"""Tests de l'injection de défauts et de la robustesse de base de la chaîne tracker + AEB (sans CARLA)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusion.adas import AEB, AEBConfig, FaultInjector, Tracker


def test_fantome_dure_ghost_len_images_et_se_rapproche():
    fi = FaultInjector(ghost_prob=1.0, ghost_len=3, ghost_range=(30.0, 30.0), seed=0)
    out = [fi.apply([], 10.0, 0.05) for _ in range(3)]
    assert [len(o) for o in out] == [1, 1, 1]
    assert abs(out[0][0][0] - 30.0) < 1e-9 and abs(out[1][0][0] - 29.5) < 1e-9 and abs(out[2][0][0] - 29.0) < 1e-9
    assert fi.n_ghost_frames == 3


def test_sans_defaut_ne_modifie_rien():
    fi = FaultInjector()
    assert not fi.active and fi.apply([(20.0, 0.1)], 10.0, 0.05) == [(20.0, 0.1)]


def test_perte_totale_supprime_les_mesures():
    assert FaultInjector(drop_prob=1.0).apply([(20.0, 0.0)], 10.0, 0.05) == []


def _freine(ghost_len, seed, secs=30.0, ghost_prob=0.01):
    dt, v = 0.05, 50 / 3.6
    tr, aeb = Tracker(dt, model="ca", jerk=2.0), AEB(AEBConfig(use_lead_decel=True), dt)
    fi = FaultInjector(ghost_prob, ghost_len, seed=seed)
    for _ in range(int(secs / dt)):
        if aeb.step(tr.step(fi.apply([], v, dt)), v).brake > 0:
            return True
    return False


def test_une_fausse_detection_isolee_ne_declenche_pas_le_freinage():
    """Propriété de protection : un fantôme d'une seule image est rejeté (route dégagée, 50 km/h, ~6 fantômes par essai)."""
    assert not any(_freine(1, s) for s in range(20))
