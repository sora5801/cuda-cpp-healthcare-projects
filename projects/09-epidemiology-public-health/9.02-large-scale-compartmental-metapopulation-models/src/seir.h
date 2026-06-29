// ===========================================================================
// src/seir.h  --  Shared (host + device) SEIR model + RK4 integrator
// ---------------------------------------------------------------------------
// Project 9.02 : Large-Scale Compartmental & Metapopulation Models
//
// WHAT THIS PROJECT COMPUTES
//   An ENSEMBLE of SEIR epidemic trajectories: the same compartmental ODE solved
//   for thousands of different parameter sets (a sweep of transmission rate beta
//   and recovery rate gamma). Each parameter set is an INDEPENDENT ODE solve, so
//   each GPU thread integrates one trajectory -- the natural mapping for Monte
//   Carlo / uncertainty-quantification ensembles.
//
//   The model (constant population N = S+E+I+R):
//     dS/dt = -beta*S*I/N
//     dE/dt =  beta*S*I/N - sigma*E      (E = exposed/latent)
//     dI/dt =  sigma*E    - gamma*I      (I = infectious)
//     dR/dt =  gamma*I
//   beta = transmission rate, 1/sigma = latent period, 1/gamma = infectious
//   period. The basic reproduction number is R0 = beta/gamma.
//
//   The derivative AND the RK4 step live here as __host__ __device__ inline
//   functions so the CPU reference and the GPU kernel integrate IDENTICALLY ->
//   their results match to round-off. SEIR_HD expands to __host__ __device__
//   under nvcc, nothing under the host compiler.
// ===========================================================================
#pragma once

#ifdef __CUDACC__
#define SEIR_HD __host__ __device__
#else
#define SEIR_HD
#endif

// SEIR right-hand side: given the state, write the four time-derivatives.
SEIR_HD inline void seir_deriv(double S, double E, double I, double R,
                              double N, double beta, double sigma, double gamma,
                              double& dS, double& dE, double& dI, double& dR) {
    const double infection = beta * S * I / N;   // new infections per unit time
    dS = -infection;
    dE = infection - sigma * E;
    dI = sigma * E - gamma * I;
    dR = gamma * I;
    (void)R;                                      // R does not appear on the RHS
}

// One classical 4th-order Runge-Kutta step (dt) advancing the state in place.
// RK4 evaluates the derivative at four points and combines them; it is accurate
// (O(dt^4) error) and stable for these smooth epidemic dynamics.
SEIR_HD inline void rk4_step(double& S, double& E, double& I, double& R,
                            double N, double beta, double sigma, double gamma, double dt) {
    double k1S, k1E, k1I, k1R;
    double k2S, k2E, k2I, k2R;
    double k3S, k3E, k3I, k3R;
    double k4S, k4E, k4I, k4R;

    seir_deriv(S, E, I, R, N, beta, sigma, gamma, k1S, k1E, k1I, k1R);
    seir_deriv(S + 0.5*dt*k1S, E + 0.5*dt*k1E, I + 0.5*dt*k1I, R + 0.5*dt*k1R,
               N, beta, sigma, gamma, k2S, k2E, k2I, k2R);
    seir_deriv(S + 0.5*dt*k2S, E + 0.5*dt*k2E, I + 0.5*dt*k2I, R + 0.5*dt*k2R,
               N, beta, sigma, gamma, k3S, k3E, k3I, k3R);
    seir_deriv(S + dt*k3S, E + dt*k3E, I + dt*k3I, R + dt*k3R,
               N, beta, sigma, gamma, k4S, k4E, k4I, k4R);

    S += (dt / 6.0) * (k1S + 2.0*k2S + 2.0*k3S + k4S);
    E += (dt / 6.0) * (k1E + 2.0*k2E + 2.0*k3E + k4E);
    I += (dt / 6.0) * (k1I + 2.0*k2I + 2.0*k3I + k4I);
    R += (dt / 6.0) * (k1R + 2.0*k2R + 2.0*k3R + k4R);
}

// Per-trajectory summary outputs the analysis cares about.
struct MemberResult {
    double peak_I_frac;   // maximum infectious fraction I/N over the run
    int    peak_step;     // step index at which the peak occurred
    double attack_rate;   // final R/N  (fraction ever infected)
};

// Integrate ONE ensemble member to completion and fill its summary. Shared by
// the CPU reference and the GPU kernel.
SEIR_HD inline MemberResult integrate_member(double N, double S0, double E0, double I0, double R0,
                                            double beta, double sigma, double gamma,
                                            double dt, int steps) {
    double S = S0, E = E0, I = I0, R = R0;
    double peak = I;
    int peak_step = 0;
    for (int s = 1; s <= steps; ++s) {
        rk4_step(S, E, I, R, N, beta, sigma, gamma, dt);
        if (I > peak) { peak = I; peak_step = s; }
    }
    MemberResult out;
    out.peak_I_frac = peak / N;
    out.peak_step = peak_step;
    out.attack_rate = R / N;
    return out;
}
