from scipy import special as sp
from math import sqrt, exp
from details.RegHQAMGenerator import RegHQAMGenerator
from details.Detector import MLD, ThrassosDetector
from details.Point import Point
from matplotlib import pyplot as plt
import random
import numpy as np

STEPS = 10000


class Simulation:
    def __init__(self, orders, d_min, max_snr_db, detection_methods):
        self.orders = orders
        self.d_min = d_min
        self.snr_values = self.find_snr_values(max_snr_db)
        self.detection_methods = detection_methods
        self.sep = self.initialize_sep()

    def initialize_sep(self):
        sep = dict()
        for detection_method in self.detection_methods:
            sep[detection_method] = np.ones(
                (len(self.orders), len(self.snr_values)))
        return sep

    def find_snr_values(self, max_snr_db):
        return list(range(max_snr_db))

    def pick_point(self, points):
        return random.choice(list(points))

    def add_gaussian_noise(self, point, n0):
        return Point(point.x + np.random.normal(loc=0.0, scale=sqrt(n0/2)), point.y + np.random.normal(loc=0.0, scale=sqrt(n0/2)))

    def simulate_single(self, snr, d_min, points, energy):
        n0 = energy / (10**(snr / 10))  # n0 must not be in db

        sep = dict()
        detector = dict()
        for detection_method in self.detection_methods:
            sep[detection_method] = 0
            detector[detection_method] = detection_method(
                points, d_min)

        for step in range(STEPS):
            point = self.pick_point(points)
            point_after_noise = self.add_gaussian_noise(point, n0)
            for detection_method in self.detection_methods:
                correct_detection = detector[detection_method].bool_detect(
                    point, point_after_noise)
                sep[detection_method] = sep[detection_method] + \
                    (1 / (step + 1)) * \
                    ((not correct_detection).real - sep[detection_method])
        return sep

    def simulate(self):
        for i in range(len(self.orders)):
            gen = RegHQAMGenerator(self.d_min)
            points = gen.generate(self.orders[i])
            es = gen.compute_av_energy(points)

            for j in range(len(self.snr_values)):
                print(
                    "order = " + str(self.orders[i]) + "  Es/N0 = " + str(self.snr_values[j]))
                cur_sep = self.simulate_single(
                    self.snr_values[j], gen.d_min, points, es)
                for detection_method in self.detection_methods:
                    self.sep[detection_method][i][j] = cur_sep[detection_method]

        for detection_method in self.detection_methods:
            np.savetxt("sep_reg" + detection_method([], None).name,
                       self.sep[detection_method])

    def compute_nrmse(self):
        b = dict()
        b[16] = 10.0
        b[32] = 13.0
        b[64] = 22.0
        b[128] = 27.0
        b[256] = 46.0
        b[512] = 64.0
        b[1024] = 128.0

        k = dict()
        k[16] = 0.8711505
        k[32] = 0.7233274
        k[64] = 0.5222431
        k[128] = 0.5088351
        k[256] = 0.3936315
        k[512] = 0.3672311
        k[1024] = 0.2982858

        a = dict()
        a[16] = 9.0
        a[32] = 17.75
        a[64] = 37.0
        a[128] = 72.0
        a[256] = 149.0
        a[512] = 289.06
        a[1024] = 597.0

        thrassos = np.ones((len(self.orders), len(self.snr_values)))
        rugini = np.ones((len(self.orders), len(self.snr_values)))
        for i in range(len(self.orders)):
            M = self.orders[i]
            for j in range((len(self.snr_values))):
                w = (10**(self.snr_values[j] / 10))
                rugini[i][j] = rugini_approx(M, w)
                thrassos[i][j] = thrassos_approx(
                    M, w, a[self.orders[i]], b[self.orders[i]], k[self.orders[i]])

        metric = dict()
        for i in range(len(self.orders)):
            thr = nrmse(self.sep[MLD][i], thrassos[i])
            rug = nrmse(self.sep[MLD][i], rugini[i])
            metric[self.orders[i]] = {"thrassos": thr, "rugini": rug}
        return metric

    def plot_approx(self):
        b = dict()
        b[16] = 10.0
        b[32] = 13.0
        b[64] = 22.0
        b[128] = 27.0
        b[256] = 46.0
        b[512] = 64.0
        b[1024] = 128.0

        k = dict()
        k[16] = 0.8711505
        k[32] = 0.7233274
        k[64] = 0.5222431
        k[128] = 0.5088351
        k[256] = 0.3936315
        k[512] = 0.3672311
        k[1024] = 0.2982858

        a = dict()
        a[16] = 9.0
        a[32] = 17.75
        a[64] = 37.0
        a[128] = 72.0
        a[256] = 149.0
        a[512] = 289.06
        a[1024] = 597.0

        x = np.linspace(0, 40)
        thrassos = np.ones((len(self.orders), len(x)))
        rugini = np.ones((len(self.orders), len(x)))
        for i in range(len(self.orders)):
            M = self.orders[i]
            for j in range(len(x)):
                w = (10**(x[j] / 10))
                rugini[i][j] = rugini_approx(M, w)
                thrassos[i][j] = thrassos_approx(
                    M, w, a[self.orders[i]], b[self.orders[i]], k[self.orders[i]])

        fig, ax = plt.subplots()
        for i in range(len(self.orders)):
            ax.plot(self.snr_values, self.sep[MLD][i],
                    label=f"{self.orders[i]}-HQAM (sim.)", linestyle="", marker='o')
            if i == len(self.orders) - 1:
                label_1 = "M-HQAM (approx.)"
                label_2 = "M-HQAM ([8] approx.)"
            else:
                label_1 = ""
                label_2 = ""
            ax.plot(x, thrassos[i], label=label_1, color='k')
            ax.plot(x, rugini[i], label=label_2, linestyle='--', color='gray')

        ax.set_yscale('log')
        ax.set_ylim([1e-05, 1])
        # ax.legend()
        ax.set(xlabel="Es/N0 (dB)", ylabel="Symbol Error Probability")
        fig.savefig("approximations.png")
        plt.show()
        plt.close()

    def plot_upper_bounds(self):
        a = dict()
        a[16] = 9.0
        a[32] = 17.75
        a[64] = 37.0
        a[128] = 72.0
        a[256] = 149.0
        a[512] = 289.06
        a[1024] = 597.0

        b = dict()
        b[16] = 10.0
        b[32] = 13.0
        b[64] = 22.0
        b[128] = 27.0
        b[256] = 46.0
        b[512] = 64.0
        b[1024] = 128.0

        x = np.linspace(0, 40)
        upper_1 = np.ones((len(self.orders), len(x)))
        upper_2 = np.ones((len(self.orders), len(x)))
        for i in range(len(self.orders)):
            M = self.orders[i]
            for j in range(len(x)):
                w = (10**(x[j] / 10))
                upper_1[i][j] = thrassos_approx(
                    M, w, a[self.orders[i]], b[self.orders[i]], 0.0)
                upper_2[i][j] = thrassos_approx(
                    M, w, a[self.orders[i]], 0.0, 0.0)

        fig, ax = plt.subplots()
        for i in range(len(self.orders)):
            ax.plot(self.snr_values, self.sep[MLD][i],
                    label=f"{self.orders[i]}-HQAM (sim.)", linestyle="", marker='o')
            if i == len(self.orders) - 1:
                label_1 = "upper bound (remark 1)"
                label_2 = "upper bound (remark 2)"
            else:
                label_1 = ""
                label_2 = ""
            ax.plot(x, upper_1[i], label=label_1, color='k')
            ax.plot(x, upper_2[i], label=label_2, linestyle='--', color='gray')

        ax.set_yscale('log')
        ax.set_ylim([1e-05, 1])
        # ax.legend()
        ax.set(xlabel="Es/N0 (dB)", ylabel="Symbol Error Probability")
        fig.savefig("upper_bounds.png")
        plt.show()
        plt.close()

    def plot_detection_methods(self):
        fig, ax = plt.subplots()
        for i in range(len(self.orders)):
            if i == len(self.orders) - 1:
                label_1 = "proposed detection"
            else:
                label_1 = ""
            ax.plot(self.snr_values, self.sep[MLD][i],
                    label=f"MLD {self.orders[i]}-HQAM (sim.)", linestyle="", marker='o')
            ax.plot(self.snr_values, self.sep[ThrassosDetector]
                    [i], label=label_1, linestyle='-', color='k')

        ax.set_yscale('log')
        ax.set_ylim([1e-05, 1])
        # ax.legend()
        ax.set(xlabel="Es/N0 (dB)", ylabel="Symbol Error Rate")
        fig.savefig("detection_methods.png")
        plt.show()
        plt.close()


def qfunc(x):
    return 0.5-0.5*sp.erf(x/sqrt(2))


def thrassos_approx(M, w, a, b, k):
    l1 = (4 * k**2) / (3 * a)
    l2 = (4 * k * (1 - k)) / (sqrt(3) * a)
    l3 = (1 - k)**2 / a
    return ((2*M - b) / (2*M)) * exp(-w*(l1 + l2 + l3)) + (b / M) * qfunc(sqrt(2 * w * l1) + sqrt(2 * w * l3))


def rugini_approx(M, w):
    a = 24.0 / (7*M - 4)
    K = 2 * (3 - 4 * M**(-1/2) + M**(-1))
    Kc = 6 * (1 - M**(-1/2)) ** 2
    return K * qfunc(sqrt(a * w)) + (2/3) * Kc * qfunc(sqrt(2 * a * w / 3)) ** 2 - 2 * Kc * qfunc(sqrt(a * w)) * qfunc(sqrt(a * w / 3))


def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted


def mse(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    return rmse(actual, predicted) / (actual.max() - actual.min())


if __name__ == '__main__':
    simulation = Simulation([16, 32, 64, 128, 256, 512, 1024], 1, 40, [
                            MLD, ThrassosDetector])
    simulation.sep[MLD] = np.loadtxt("sep_regMLD")
    simulation.sep[ThrassosDetector] = np.loadtxt("sep_regThrassos' method")
    # simulation.simulate()
    simulation.plot_approx()
    simulation.plot_upper_bounds()
    simulation.plot_detection_methods()
    print(simulation.compute_nrmse())
