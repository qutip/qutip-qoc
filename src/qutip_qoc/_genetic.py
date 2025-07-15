import numpy as np
import qutip as qt
import time
from qutip_qoc.result import Result

class _GENETIC:
    def __init__(
        self,
        objectives,
        control_parameters,
        time_interval,
        time_options,
        alg_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs,
        qtrl_optimizers,
    ):
        self._objective = objectives[0]
        self._Hd = self._objective.H[0]
        self._Hc_lst = [H[0] if isinstance(H, list) else H for H in self._objective.H[1:]]
        self._initial = self._objective.initial
        self._target = self._objective.target
        self._norm_fac = 1 / self._target.norm()

        self._evo_time = time_interval.evo_time
        self.N_steps = time_interval.n_tslots
        self.N_controls = len(self._Hc_lst)
        self.N_var = self.N_controls * self.N_steps

        self._alg_kwargs = alg_kwargs
        self.N_pop = alg_kwargs.get("population_size", 100)
        self.generations = alg_kwargs.get("generations", 100)
        self.mutation_rate = alg_kwargs.get("mutation_rate", 0.3)
        self.fid_err_targ = alg_kwargs.get("fid_err_targ", 1e-4)
        self._stagnation_patience = 20  # Internally fixed

        self._integrator_kwargs = integrator_kwargs
        self._fid_type = alg_kwargs.get("fid_type", "PSU")

        self._generator = self._prepare_generator()
        self._solver = qt.MESolver(H=self._generator, options=self._integrator_kwargs) \
            if self._Hd.issuper else qt.SESolver(H=self._generator, options=self._integrator_kwargs)

        self._result = Result(
            objectives=[self._objective],
            time_interval=time_interval,
            start_local_time=time.time(),
            guess_controls=[self._generator],
            guess_params=control_parameters.get("guess"),
            var_time=False,
            qtrl_optimizers=qtrl_optimizers,
        )
        self._result.iter_seconds = []
        self._result.infidelity = np.inf
        self._result._final_states = []

    def _prepare_generator(self):
        args = {f"p{i+1}_{j}": 0.0 for i in range(self.N_controls) for j in range(self.N_steps)}

        def make_coeff(i, j):
            return lambda t, args: args[f"p{i+1}_{j}"] if int(t / (self._evo_time / self.N_steps)) == j else 0

        H_qev = [self._Hd]
        for i, Hc in enumerate(self._Hc_lst):
            for j in range(self.N_steps):
                H_qev.append([Hc, make_coeff(i, j)])

        return qt.QobjEvo(H_qev, args=args)

    def _infid(self, params):
        args = {f"p{i+1}_{j}": params[i * self.N_steps + j] for i in range(self.N_controls) for j in range(self.N_steps)}
        result = self._solver.run(self._initial, [0.0, self._evo_time], args=args)
        final_state = result.final_state
        self._result._final_states.append(final_state)

        if self._fid_type == "TRACEDIFF":
            diff = final_state - self._target
            fid = 0.5 * np.real((diff.dag() * diff).tr())
        else:
            overlap = self._norm_fac * self._target.overlap(final_state)
            fid = 1 - np.abs(overlap) if self._fid_type == "PSU" else 1 - np.real(overlap)

        return fid

    def step(self, params):
        t0 = time.time()
        val = -self._infid(params)
        self._result.iter_seconds.append(time.time() - t0)
        return val

    def initial_population(self):
        return np.random.uniform(-1, 1, (self.N_pop, self.N_var))

    def darwin(self, population, fitness):
        indices = np.argsort(-fitness)[:self.N_pop // 2]
        return population[indices], fitness[indices]

    def pairing(self, survivors, survivor_fitness):
        prob_dist = survivor_fitness - np.min(survivor_fitness)
        prob_dist /= np.sum(prob_dist)

        mothers = np.random.choice(len(survivors), size=self.N_pop // 4, p=prob_dist)
        fathers = []
        for m in mothers:
            p = prob_dist.copy()
            p[m] = 0
            p /= np.sum(p)
            fathers.append(np.random.choice(len(survivors), p=p))
        return survivors[mothers], survivors[fathers]

    def mating_procedure(self, ma, da):
        beta = np.random.rand(*ma.shape)
        swap = np.random.randint(0, 2, ma.shape)
        beta_inv = 1 - beta

        new1 = swap * beta * ma + swap * beta_inv * da + (1 - swap) * ma
        new2 = swap * beta * da + swap * beta_inv * ma + (1 - swap) * da
        return np.vstack((new1, new2))

    def build_next_gen(self, survivors, offspring):
        return np.vstack((survivors, offspring))

    def mutate(self, population):
        n_mut = int((population.shape[0] - 1) * population.shape[1] * self.mutation_rate)
        row = np.random.randint(1, population.shape[0], size=n_mut)
        col = np.random.randint(0, population.shape[1], size=n_mut)
        population[row, col] += np.random.normal(0, 0.3, size=n_mut)
        population[row, col] = np.clip(population[row, col], -2, 2)
        return population

    def optimize(self):
        population = self.initial_population()
        best_fit = -np.inf
        best_chrom = None
        history = []
        no_improvement_counter = 0

        for gen in range(self.generations):
            fitness = np.array([self.step(chrom) for chrom in population])
            max_fit = np.max(fitness)
            history.append(max_fit)

            if max_fit > best_fit:
                best_fit = max_fit
                best_chrom = population[np.argmax(fitness)]
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            self._result.infidelity = min(self._result.infidelity, -max_fit)

            if -max_fit <= self.fid_err_targ:
                break

            if no_improvement_counter >= self._stagnation_patience:
                break

            survivors, survivor_fit = self.darwin(population, fitness)
            mothers, fathers = self.pairing(survivors, survivor_fit)
            offspring = self.mating_procedure(mothers, fathers)
            population = self.build_next_gen(survivors, offspring)
            population = self.mutate(population)

        self._result.optimized_params = best_chrom.tolist()
        self._result.infidelity = -best_fit
        self._result.end_local_time = time.time()
        self._result.n_iters = len(history)
        self._result.new_params = self._result.optimized_params
        self._result._optimized_controls = best_chrom.tolist()
        self._result._optimized_H = [self._generator]
        self._result._final_states = self._result._final_states  # expose final_states
        self._result.guess_params = self._result.guess_params or []
        self._result.var_time = False

        self._result.message = (
            f"Stopped early: reached infidelity target {self.fid_err_targ}"
            if -best_fit <= self.fid_err_targ else
            f"Stopped due to stagnation after {self._stagnation_patience} generations"
            if no_improvement_counter >= self._stagnation_patience else
            "Optimization completed successfully"
        )
        return self._result
    
    def result(self):
        self._result.start_local_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self._result.start_local_time)
        )
        self._result.end_local_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self._result.end_local_time)
        )
        return self._result



from qutip_qoc import Objective, _TimeInterval

initial = qt.basis(2, 0)
target = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
H_d = qt.sigmaz()
H_c = [qt.sigmax()]

objective = Objective(H=[H_d] + H_c, initial=initial, target=target)
time_interval = _TimeInterval(evo_time=1.0, n_tslots=20)

control_parameters = {"guess": [0.0] * 20}
alg_kwargs = {
    "population_size": 50,
    "generations": 100,
    "mutation_rate": 0.3,
    "fid_err_targ": 1e-3
}
integrator_kwargs = {"rtol": 1e-5, "atol": 1e-6}

ga = _GENETIC(objectives=[objective],
              control_parameters=control_parameters,
              time_interval=time_interval,
              time_options=None,
              alg_kwargs=alg_kwargs,
              optimizer_kwargs=None,
              minimizer_kwargs=None,
              integrator_kwargs=integrator_kwargs,
              qtrl_optimizers=None)

result = ga.optimize()
print("Best fidelity:", 1 - result.infidelity)
