#include <mujoco/mujoco.h>
#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

static void die(const char* msg) 
{ 
	std::fprintf(stderr, "%s\n", msg); std::exit(1); 
}

int main(int argc, char** argv) 
{
	if (argc < 2) { std::printf("Usage: %s model.xml [config.yaml]\n", argv[0]); return 1; }

	// Load model
	char err[1024] = {0};
	mjModel* m = mj_loadXML(argv[1], nullptr, err, sizeof(err));
	if (!m) { std::fprintf(stderr, "mj_loadXML error: %s\n", err); return 2; }
	mjData* d = mj_makeData(m);

	// SDL init
	if (SDL_Init(SDL_INIT_VIDEO) != 0) die(SDL_GetError());

	// Ask for desktop OpenGL compatibility context (key point)
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_Window* win = SDL_CreateWindow("MuJoCo (SDL2 OpenGL compat)",
										SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
										1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
	if (!win) die(SDL_GetError());

	SDL_GLContext glctx = SDL_GL_CreateContext(win);
	if (!glctx) die(SDL_GetError());

	SDL_GL_SetSwapInterval(0);

	// Print GL info
	std::printf("GL_VERSION : %s\n", glGetString(GL_VERSION));
	std::printf("GL_VENDOR  : %s\n", glGetString(GL_VENDOR));
	std::printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));

	// MuJoCo visualization
	mjvCamera cam;   mjv_defaultCamera(&cam);
	mjvOption opt;   mjv_defaultOption(&opt);
	mjvScene scn;    mjv_defaultScene(&scn);
	mjrContext con;  mjr_defaultContext(&con);

	mjv_makeScene(m, &scn, 2000);
	mjr_makeContext(m, &con, mjFONTSCALE_150);   // will error if context isn't compatible :contentReference[oaicite:4]{index=4}

	// Controller entries (filled if YAML provided)
	struct JointCtrl { int jid; int dofadr; int qposadr; double kp; double kd; double target; bool has_limit; double lim_lo; double lim_hi; };
	std::vector<JointCtrl> controllers;

	// Parse YAML config if provided as argv[2]
	if (argc >= 3) {
		try {
			YAML::Node cfg = YAML::LoadFile(argv[2]);
			YAML::Node pd = cfg["pd"];
			bool per_joint = pd["per_joint_kp_kd"].as<bool>(true);
			double global_kp = pd["kp"].as<double>(220.0);
			double global_kd = pd["kd"].as<double>(10.0);
			double global_qd_des = pd["qd_des"].as<double>(0.0);
			double global_tau_ff = pd["tau_ff"].as<double>(0.0);

			YAML::Node targets = cfg["targets"];
			YAML::Node joints = cfg["joints"];
			YAML::Node limits = cfg["limits"];

			if (targets && targets.IsMap()) {
				for (auto it = targets.begin(); it != targets.end(); ++it) {
					std::string name = it->first.as<std::string>();
					double targ = it->second.as<double>(0.0);

					int jid = mj_name2id(m, mjOBJ_JOINT, name.c_str());
					if (jid < 0) {
						std::cerr << "Warning: joint '" << name << "' not found in model\n";
						continue;
					}

					JointCtrl jc;
					jc.jid = jid;
					jc.dofadr = m->jnt_dofadr[jid];
					jc.qposadr = m->jnt_qposadr[jid];
					jc.target = targ;
					jc.has_limit = false;
					jc.lim_lo = -1e9; jc.lim_hi = 1e9;

					// kp/kd selection
					if (per_joint && joints && joints[name] && joints[name].IsMap()) {
						jc.kp = joints[name]["kp"].as<double>(global_kp);
						jc.kd = joints[name]["kd"].as<double>(global_kd);
					} else {
						jc.kp = global_kp;
						jc.kd = global_kd;
					}

					// optional limits
					if (limits && limits["enabled"].as<bool>(false) && limits[name] && limits[name].IsSequence()) {
						auto seq = limits[name];
						if (seq.size() >= 2) {
							jc.lim_lo = seq[0].as<double>();
							jc.lim_hi = seq[1].as<double>();
							jc.has_limit = true;
						}
					}

					// store additional feedforward/qd if present per-pd block
					// For simplicity we use global qd_des and tau_ff for all joints here

					controllers.push_back(jc);
				}
			}
		} catch (const std::exception &e) {
			std::cerr << "Failed to load YAML config: " << e.what() << "\n";
		}
	}

	// set initial positions from controllers (apply YAML targets as initial qpos)
	if (!controllers.empty()) {
		for (const auto &jc : controllers) {
			if (jc.qposadr >= 0) d->qpos[jc.qposadr] = jc.target;
			if (jc.dofadr >= 0) d->qvel[jc.dofadr] = 0.0;
		}
		mj_forward(m, d);
	}

	bool running = true;
	while (running) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
		if (e.type == SDL_QUIT) running = false;
		if (e.type == SDL_KEYDOWN && (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_ESCAPE))
			running = false;
		}

		// zero any previous applied joint torques
		for (int i = 0; i < m->nv; ++i) d->qfrc_applied[i] = 0.0;

		// apply PD torques from controllers
		if (!controllers.empty()) {
			for (const auto &jc : controllers) {
				double q = d->qpos[jc.qposadr];
				double qvel = d->qvel[jc.dofadr];
				double qd_des_local = 0.0; // default (can extend to read per-joint)
				double tau_ff_local = 0.0;

				double torque = jc.kp * (jc.target - q) + jc.kd * (qd_des_local - qvel) + tau_ff_local;
				if (jc.has_limit) {
					if (torque < jc.lim_lo) torque = jc.lim_lo;
					if (torque > jc.lim_hi) torque = jc.lim_hi;
				}
				d->qfrc_applied[jc.dofadr] += torque;
			}
		}

		mj_step(m, d);

		int W, H;
		SDL_GL_GetDrawableSize(win, &W, &H);
		mjrRect vp = {0, 0, W, H};

		mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
		mjr_render(vp, &scn, &con);

		SDL_GL_SwapWindow(win);
	}

	mjr_freeContext(&con);
	mjv_freeScene(&scn);
	mj_deleteData(d);
	mj_deleteModel(m);

	SDL_GL_DeleteContext(glctx);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}