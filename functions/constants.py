# ==========================
# Dizionari centralizzati
# ==========================
classi_rischio = {
    "Frane": {
        "Molto elevata P4": (0.05, 20),
        "Elevata P3": (0.05, 50),
        "Media P2": (0.02, 100),
        "Moderata P1": (0.02, 200),
        "Aree di Attenzione AA": (0.01, 500),
        "Molto bassa": (0.01, 500)
    },
    "Idro": {
        "Pericolosità idraulica elevata - HighProbabilityHazard": (0.2, 20, 50),
        "Pericolosità idraulica media - MediumProbabilityHazard": (0.1, 100, 200),
        "Pericolosità idraulica bassa - LowProbabilityHazard": (0.04, 200, 500)
    }
}

alpha_tilde_classi_frane = {
    "Molto elevata P4": 8,
    "Elevata P3": 6,
    "Media P2": 4,
    "Moderata P1": 2,
    "Aree di Attenzione AA": 2,
    "Molto bassa": 2
}
