import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


class EigSolver:
    def __init__(
        self,
        n: int = 25,
        beta: float = 1.0,
        x1: float | None = None,
        equation_type: str = "gamma2-nonlinear"
    ) -> None:
        self.n = n
        self.beta = beta

        self.alpha = beta + 1 / (n ** 2 + 10)
        self.p = 3 + 1 / (n + 1)
        self.h = 5 + ((-1) ** n) / (n + 3)

        self.x1 = self.h if x1 is None else x1

        self.u_0 = 0
        self.du_0 = (n + 2) / n + n / (n ** 2 + 1)

        self.u_h = self.h

        self.equation_type = equation_type

    def eps(self, x: float) -> float:
        return (1 + 1 / self.n) * x + (x ** 2) / (self.n ** 3 + 1)

    def rhs_equation(self, x: float, u: list[float], gamma: float) -> list[float]:
        # equation_type: "gamma2-nonlinear", "gamma-nonlinear", "gamma2-linear", "gamma-linear"
        if self.equation_type == "gamma2-nonlinear":
            return [
                u[1],
                -(self.eps(x) - gamma ** 2) * u[0] - self.alpha * np.abs(u[0]) ** self.p
            ]
        elif self.equation_type == "gamma-nonlinear":
            return [
                u[1],
                -(self.eps(x) - gamma) * u[0] - self.alpha * np.abs(u[0]) ** self.p
            ]
        elif self.equation_type == "gamma2-linear":
            return [
                u[1],
                -(self.eps(x) - gamma ** 2) * u[0]
            ]
        elif self.equation_type == "gamma-linear":
            return [
                u[1],
                -(self.eps(x) - gamma) * u[0]
            ]
        else:
            return [
                u[1],
                -(self.eps(x) - gamma ** 2) * u[0] - self.alpha * np.abs(u[0]) ** self.p
            ]

    def shoot(self, gamma: float) -> float:
        sol = solve_ivp(
            self.rhs_equation,
            [0, self.x1],
            [self.u_0, self.du_0],
            args=(gamma,),
            t_eval=[self.u_h],
            dense_output=True
        )

        if not sol.success or sol.y.shape[1] == 0:
            return np.nan

        return sol.y[0][-1]

    def find_eigenvalues(
        self,
        eigen_count: int = 5,
        gamma_start: float = 0.1,
        gamma_end: float = 10.0,
        gamma_count: float | None = None,
        verbose: bool = True
    ) -> list[float]:
        gamma_count = int(((gamma_end - gamma_start) / 2) * 100) if gamma_count is None else gamma_count

        gammas = np.linspace(gamma_start, gamma_end, gamma_count)
        uhs = [self.shoot(g) for g in gammas]

        eigenvalues = []

        for i in range(1, len(gammas)):
            if np.sign(uhs[i - 1]) != np.sign(uhs[i]):
                a, b = gammas[i - 1], gammas[i]

                if any(abs(a - ev) < 1e-5 or abs(b - ev) < 1e-5 for ev in eigenvalues):
                    continue
                try:
                    res = root_scalar(
                        self.shoot,
                        bracket=[a, b],
                        method="brentq",
                        xtol=1e-7
                    )

                    if res.converged:
                        gamma_root = res.root

                        if all(abs(gamma_root - ev) > 1e-5 for ev in eigenvalues):
                            eigenvalues.append(gamma_root)

                            if verbose:
                                print(f"Найдено СЗ ({i} из {len(gammas)}) gamma = {gamma_root:.8f}")

                            if len(eigenvalues) >= eigen_count:
                                break
                except Exception:
                    continue

        return eigenvalues

    def fast_find_eigenvalues(
        self,
        eigen_count: int = 5,
        gamma_start: float = 0.1,
        gamma_end: float = 10.0,
        gamma_step: float = 0.01,
        verbose: bool = True
    ) -> list[float]:
        """
        Быстрый поиск собственных значений:
        1. Грубый проход по сетке с шагом gamma_step.
        2. Локальное уточнение корней только в интервалах смены знака.
        """
        # 1. Грубая сетка
        gamma_points = int(np.ceil((gamma_end - gamma_start) / gamma_step)) + 1
        gammas = np.linspace(gamma_start, gamma_end, gamma_points)
        uhs = [self.shoot(g) for g in gammas]

        eigenvalues = []
        for i in range(1, len(gammas)):
            if np.isnan(uhs[i - 1]) or np.isnan(uhs[i]):
                continue
            if np.sign(uhs[i - 1]) != np.sign(uhs[i]):
                a, b = gammas[i - 1], gammas[i]
                # Уточняем корень через root_scalar только в найденном интервале
                try:
                    res = root_scalar(
                        self.shoot,
                        bracket=[a, b],
                        method="brentq",
                        xtol=1e-7
                    )
                    if res.converged:
                        gamma_root = res.root
                        # Проверяем, не найден ли уже этот корень
                        if all(abs(gamma_root - ev) > 1e-5 for ev in eigenvalues):
                            eigenvalues.append(gamma_root)
                            if verbose:
                                print(f"Найдено СЗ gamma = {gamma_root:.8f}")
                            if len(eigenvalues) >= eigen_count:
                                return eigenvalues
                except Exception:
                    continue

        return eigenvalues

    def adaptive_find_eigenvalues(
        self,
        eigen_count=5,
        gamma_start=1.0,
        gamma_end=10.0,
        min_interval=1e-5,
        gamma_step=0.1,
        verbose=True,
        max_depth=10,
        patience=50,
        max_points=1000
    ):
        found_eigenvalues = []
        search_eps = 1e-5
        intervals = [(gamma_start, gamma_end, 0)]  # (a, b, depth)
        attempts = 0

        def is_close_to_found(val):
            return any(abs(val - ev) < search_eps for ev in found_eigenvalues)

        while intervals and attempts < patience:
            a, b, depth = intervals.pop(0)
            
            attempts += 1

            if b - a < min_interval or depth > max_depth:
                continue

            # Если оба конца интервала уже близки к найденным СЗ, пропускаем
            if is_close_to_found(a) and is_close_to_found(b):
                continue

            # Адаптивное количество точек: больше точек для поиска большего числа СЗ
            gamma_points = min(max_points, max(10, int(10 * eigen_count), int((b - a) / gamma_step)))
            gammas = np.linspace(a, b, gamma_points)
            uhs = [self.shoot(g) for g in gammas]

            # Проверка на NaN и константность
            if all(np.isnan(u) for u in uhs):
                continue
            if np.nanmax(uhs) - np.nanmin(uhs) < 1e-12:
                continue

            i = 1
            found_in_this_interval = False
            while i < len(gammas):
                if np.isnan(uhs[i - 1]) or np.isnan(uhs[i]):
                    i += 1
                    continue
                if np.sign(uhs[i - 1]) != np.sign(uhs[i]):
                    left, right = gammas[i - 1], gammas[i]
                    # Проверяем, не нашли ли мы уже это СЗ (по центру интервала)
                    mid = 0.5 * (left + right)
                    if is_close_to_found(left) or is_close_to_found(right) or is_close_to_found(mid):
                        i += 1
                        continue
                    try:
                        res = root_scalar(
                            self.shoot,
                            bracket=[left, right],
                            method="brentq",
                            xtol=1e-7
                        )
                        if res.converged:
                            gamma_root = res.root
                            # Проверяем, не нашли ли мы уже это СЗ (с учетом точности)
                            if not is_close_to_found(gamma_root):
                                found_eigenvalues.append(gamma_root)
                                found_in_this_interval = True
                                if verbose:
                                    print(f"Найдено СЗ gamma = {gamma_root:.8f} (depth = {depth}, attempt = {attempts})")
                                # После нахождения СЗ делим интервал на две части и исследуем глубже,
                                # только если они не слишком малы и не содержат уже найденное СЗ
                                if gamma_root - left > min_interval and not is_close_to_found(left):
                                    intervals.append((left, gamma_root - min_interval, depth + 1))
                                if right - gamma_root > min_interval and not is_close_to_found(right):
                                    intervals.append((gamma_root + min_interval, right, depth + 1))
                                # Не увеличиваем i, чтобы не пропустить близкие корни
                                continue
                    except Exception:
                        pass
                i += 1

            # Если в этом интервале не найдено ни одного СЗ, разбиваем интервал пополам для дальнейшего поиска
            if not found_in_this_interval and (b - a) > min_interval * 2 and depth < max_depth:
                mid = (a + b) / 2
                # Добавляем только если не слишком близко к уже найденным
                if not is_close_to_found(a) and not is_close_to_found(mid):
                    intervals.append((a, mid, depth + 1))
                if not is_close_to_found(mid) and not is_close_to_found(b):
                    intervals.append((mid, b, depth + 1))

        found_eigenvalues = sorted(found_eigenvalues)

        if verbose:
            print(f"Поиск завершён: найдено {len(found_eigenvalues)} СЗ за {attempts} попыток.")

        return found_eigenvalues
