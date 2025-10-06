import pandas as pd
import numpy as np
from sklearn import metrics
import math
import scipy.stats as st
import warnings
from scipy.stats import gaussian_kde
import optuna
import matplotlib.pyplot as plt

def get_data_statistics(input_path):
    # Carica il file
    df = pd.read_excel(input_path)

    # Estrai l'anno
    df['Year'] = df['Date'].dt.year

    # Calcola i rendimenti logaritmici
    df['Log_Returns'] = np.log(df['GMEPIT24 Index'] / df['GMEPIT24 Index'].shift(1))

    # Crea DataFrame con rendimenti e proxy volatilità
    rendimenti_giornalieri = df[['Year', 'Date', 'Log_Returns', 'GMEPIT24 Index']].set_index('Date')
    rendimenti_giornalieri['proxy_vol'] = rendimenti_giornalieri['Log_Returns']**2

    # Salva il grafico dell'indice
    plt.figure(figsize=(10, 4))
    rendimenti_giornalieri['GMEPIT24 Index'].plot(title='Andamento PUN')
    plt.tight_layout()
    plt.savefig('PUN_Index.png')
    plt.close()

    # Salva il grafico dei log returns
    plt.figure(figsize=(10, 4))
    rendimenti_giornalieri['Log_Returns'].plot(title='Log Returns')
    plt.tight_layout()
    plt.savefig('Log_Returns.png')
    plt.close()

    # Salva il grafico della proxy volatilità
    plt.figure(figsize=(10, 4))
    rendimenti_giornalieri['proxy_vol'].plot(title='Proxy Volatility')
    plt.tight_layout()
    plt.savefig('Proxy_Volatility.png')
    plt.close()

    return df, rendimenti_giornalieri


def historical_VaR(rendimenti_giornalieri, n_simulazioni=100_000, csv_file="VaR_results.csv"):
    data = rendimenti_giornalieri['Log_Returns'].dropna()

    # --- STEP 2: Fit delle distribuzioni ---
    distributions = ['norm', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'gumbel_r', 'gennorm']
    best_fits = {}
    rmse_scores = {}

    x = np.linspace(min(data), max(data), 1000)
    hist_vals, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Istogramma e KDE
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, color='gray', label='Empirical Data')

    kde = gaussian_kde(data)
    kde_pdf = kde(x)
    plt.plot(x, kde_pdf, label="KDE", color='blue', linestyle='--')

    for dist_name in distributions:
        dist = getattr(st, dist_name)
        try:
            params = dist.fit(data)
            pdf = dist.pdf(bin_centers, *params)
            rmse = np.sqrt(np.mean((hist_vals - pdf) ** 2))
            best_fits[dist_name] = (params, pdf, rmse)
            rmse_scores[dist_name] = rmse
            plt.plot(bin_centers, pdf, label=f'{dist_name} (RMSE: {rmse:.4f})')
        except Exception:
            continue

    plt.title("Empirical vs Fitted Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Distribution_Fits.png')
    plt.close()

    # --- STEP 3: Simulazione Monte Carlo ---
    best_dist_name = min(rmse_scores, key=rmse_scores.get)
    best_dist, best_params = getattr(st, best_dist_name), best_fits[best_dist_name][0]

    simulated_returns = best_dist.rvs(*best_params, size=n_simulazioni)

    # --- STEP 4: Calcolo metriche ---
    P0 = rendimenti_giornalieri['GMEPIT24 Index'].iloc[-1]

    R_worst = np.percentile(simulated_returns, 95)
    R_medium = np.percentile(simulated_returns, 50)
    R_best = np.percentile(simulated_returns, 5)

    P_worst = P0 * np.exp(R_worst)
    P_medium = P0 * np.exp(R_medium)
    P_best = P0 * np.exp(R_best)

    # Creazione di un DataFrame per i risultati
    result_data = {
        'Percentile': ['5%', '50%', '95%'],
        'Simulated Price': [P_best, P_medium, P_worst]
    }

    results_df = pd.DataFrame(result_data)

    # Salvataggio dei risultati in un file CSV
    results_df.to_csv(csv_file, index=False)

    # Output dei risultati
    print(f"📉 Worst Case (95° percentile): Prezzo = {np.round(P_worst, 0):.4f}")
    print(f"⚖️ Medium Case (50° percentile): Prezzo = {np.round(P_medium, 0):.4f}")
    print(f"📈 Best Case (5° percentile): Prezzo = {np.round(P_best, 0):.4f}")

    return results_df  # Restituisce il DataFrame con i risultati



# Funzione di ottimizzazione dei parametri di Heston con Optuna
def optimize_heston_model(df, n_trials=2000, end_date="2027-12-31"):
    log_returns = df['Log_Returns'].dropna()

    def simulate_heston_single_path(S0, mu, kappa, theta, sigma_v, rho, days_to_simulate):
        dt = 1  # daily step
        simulated_log_returns = np.zeros(days_to_simulate)
        simulated_volatilities = np.zeros(days_to_simulate)
        v_t = theta

        for t in range(days_to_simulate):
            dW1 = np.random.randn() * np.sqrt(dt)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.randn() * np.sqrt(dt)

            v_t = np.clip(v_t + kappa * (theta - v_t) * dt + sigma_v * np.sqrt(max(v_t, 0)) * dW2, 0, 1)
            simulated_volatilities[t] = v_t
            dlogS = (mu - 0.5 * v_t) * dt + np.sqrt(v_t) * dW1
            simulated_log_returns[t] = dlogS

        return simulated_log_returns

    def objective(trial):
        sigma_long = log_returns.std()
        mu_long = log_returns.mean()

        mu = trial.suggest_float("mu", 0.000157, 0.0003)
        kappa = trial.suggest_float("kappa", 0.1, 5.0)
        theta = trial.suggest_float("theta", 0.0001, 0.01)
        sigma_v = trial.suggest_float("sigma_v", sigma_long / 10, sigma_long * 1.5)
        rho = trial.suggest_float("rho", -0.5, 0.5)

        S0 = df['GMEPIT24 Index'].iloc[-1]
        days_to_simulate = min((pd.to_datetime(end_date) - df['Date'].iloc[-1]).days, len(log_returns))

        simulated_log_returns = simulate_heston_single_path(S0, mu, kappa, theta, sigma_v, rho, days_to_simulate)
        real_log_returns = log_returns.values[-days_to_simulate:]
        rmse = np.sqrt(np.mean((real_log_returns - simulated_log_returns) ** 2))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    return best_params, study


# Funzione di simulazione Heston per n simulazioni
def simulate_heston(S0, mu, kappa, theta, sigma_v, rho, days_to_simulate, n_simulations):
    dt = 1
    simulated_log_returns = np.zeros((n_simulations, days_to_simulate))

    for i in range(n_simulations):
        v_t = theta
        for t in range(days_to_simulate):
            dW1 = np.random.randn() * np.sqrt(dt)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.randn() * np.sqrt(dt)

            v_t = np.clip(v_t + kappa * (theta - v_t) * dt + sigma_v * np.sqrt(max(v_t, 0)) * dW2, 0, 0.2)
            dlogS = (mu - 0.5 * v_t) * dt + np.sqrt(v_t) * dW1
            simulated_log_returns[i, t] = dlogS

    log_prices = np.cumsum(simulated_log_returns, axis=1)
    simulated_prices = S0 * np.exp(log_prices)
    # simulated_prices = np.clip(simulated_prices, 48, 220)

    return simulated_prices, simulated_log_returns

def run_heston(df, n_trials=2000, n_simulations=1000, end_date="2027-12-31"):
    # Step 1: Ottimizzazione dei parametri Heston tramite Optuna
    # best_params, study = optimize_heston_model(df, n_trials, end_date)

    # Step 2: Simulazione dei percorsi futuri usando i parametri ottimizzati
    S0 = df['GMEPIT24 Index'].iloc[-1]
    days_to_simulate = (pd.to_datetime(end_date) - df['Date'].iloc[-1]).days

    simulated_prices, simulated_log_returns = simulate_heston(
    S0,
    best_params["mu"],
    best_params["kappa"],
    best_params["theta"],
    best_params["sigma_v"],
    best_params["rho"],
    days_to_simulate,
    n_simulations )
    # simulated_prices = np.clip(simulated_prices, 48, 220)

    # Step 3: Visualizzare il risultato delle simulazioni
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_prices.T, color='blue', alpha=0.05)
    plt.title("Simulazione dei Percorsi Futuri (Heston Model)")
    plt.xlabel("Giorni")
    plt.ylabel("Prezzo")
    plt.grid(True)

    # Salvataggio dell'immagine
    plt.savefig('Simulazione_dei_Percorsi_Futuri_Heston_Model.png')
    plt.show()

    return best_params, simulated_prices


def get_monthly_and_yearly_distribution(df, years):
    monthly_percentiles = {}
    monthly_distributions = {}
    monthly_means = {}

    yearly_percentiles = {}
    yearly_distributions = {}
    yearly_means = {}

    for year in years:
        # ---- Mensile ----
        for month in range(1, 13):
            df_month = df[(df.index.year == year) & (df.index.month == month)]
            values = df_month.values.flatten()
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                p5, p50, p95 = np.percentile(values, [5, 50, 95])
                mean = np.mean(values)

                monthly_percentiles[(year, month)] = (p5, p50, p95)
                monthly_means[(year, month)] = mean
                monthly_distributions[(year, month)] = values
            else:
                monthly_percentiles[(year, month)] = (np.nan, np.nan, np.nan)
                monthly_means[(year, month)] = np.nan
                monthly_distributions[(year, month)] = np.array([])

        # ---- Annuale ----
        df_year = df[df.index.year == year]
        values_year = df_year.values.flatten()
        values_year = values_year[~np.isnan(values_year)]

        if len(values_year) > 0:
            p5_y, p50_y, p95_y = np.percentile(values_year, [5, 50, 95])
            mean_y = np.mean(values_year)

            yearly_percentiles[year] = (p5_y, p50_y, p95_y)
            yearly_means[year] = mean_y
            yearly_distributions[year] = values_year
        else:
            yearly_percentiles[year] = (np.nan, np.nan, np.nan)
            yearly_means[year] = np.nan
            yearly_distributions[year] = np.array([])

    return (
        monthly_distributions, monthly_percentiles, monthly_means,
        yearly_distributions, yearly_percentiles, yearly_means
    )

# Funzione per simulare distribuzioni e percentili, salvare il grafico e il CSV
def plot_and_save_distribution(sim_df, years=[2025, 2026, 2027], output_file="distribution_plot.png", csv_file_m="forecast_percentiles_monthly.csv", csv_file_y="forecast_percentiles_yearly.csv"):
    # Calcolare distribuzioni e percentili (sia mensili che annuali)
    (
        monthly_dist, monthly_percentiles, monthly_means,
        yearly_dist, yearly_percentiles, yearly_means
    ) = get_monthly_and_yearly_distribution(sim_df, years)

    # Seleziona solo i mesi con dati
    selected_months = [k for k, v in monthly_dist.items() if len(v) > 0]

    # Disegnare le distribuzioni mensili
    plt.figure(figsize=(14, 6))
    for (year, month) in selected_months:
        values = monthly_dist.get((year, month), [])
        if len(values) == 0:
            continue

        label = f"{year}-{month:02d}"
        plt.hist(values, bins=100, alpha=0.4, label=label, density=True)

        p5, p50, p95 = monthly_percentiles.get((year, month), (None, None, None))
        if p5 is not None:
            plt.axvline(p5, color='blue', linestyle='--', alpha=0.3)
        if p50 is not None:
            plt.axvline(p50, color='green', linestyle='-', alpha=0.3)
        if p95 is not None:
            plt.axvline(p95, color='red', linestyle='--', alpha=0.3)

    plt.title("Distribuzione dei prezzi simulati per mese")
    plt.xlabel("Prezzo simulato")
    plt.ylabel("Densità")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Salvare il grafico
    plt.savefig(output_file)
    plt.close()

    # Creare un DataFrame per i percentili mensili e salvarlo
    percentiles_df = pd.DataFrame.from_dict(monthly_percentiles, orient='index', columns=['5%', '50%', '95%'])
    percentiles_df.index.name = 'Anno, Mese'
    percentiles_df.dropna(inplace=True)
    percentiles_df.to_csv(csv_file_m)
    
    percentiles_df_y = pd.DataFrame.from_dict(yearly_percentiles, orient='index', columns=['5%', '50%', '95%'])
    percentiles_df_y.index.name = 'Anno'
    percentiles_df_y.dropna(inplace=True)
    percentiles_df_y.to_csv(csv_file_y)

    return monthly_percentiles, monthly_means, yearly_percentiles, yearly_means

# Funzione principale che esegue tutto
def analyze_simulation(sim_df, years):
    """
    Calcola percentili mensili e annuali senza salvare nulla su disco.
    Restituisce dizionari con percentili e medie.
    """
    (
        monthly_distributions, monthly_percentiles, monthly_means,
        yearly_distributions, yearly_percentiles, yearly_means
    ) = get_monthly_and_yearly_distribution(sim_df, years)

    # Genera il grafico ma non salva, restituisce la figura
    fig, ax = plt.subplots(figsize=(14, 6))
    selected_months = [k for k, v in monthly_distributions.items() if len(v) > 0]

    for (year, month) in selected_months:
        values = monthly_distributions.get((year, month), [])
        if len(values) == 0:
            continue
        label = f"{year}-{month:02d}"
        ax.hist(values, bins=100, alpha=0.4, label=label, density=True)
        p5, p50, p95 = monthly_percentiles.get((year, month), (None, None, None))
        if p5 is not None:
            ax.axvline(p5, color='blue', linestyle='--', alpha=0.3)
        if p50 is not None:
            ax.axvline(p50, color='green', linestyle='-', alpha=0.3)
        if p95 is not None:
            ax.axvline(p95, color='red', linestyle='--', alpha=0.3)

    ax.set_title("Distribuzione prezzi simulati per mese")
    ax.set_xlabel("Prezzo simulato")
    ax.set_ylabel("Densità")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return monthly_percentiles, monthly_means, yearly_percentiles, yearly_means, fig


def replace_last_zero_with_value(lst, last_value):
    # Trova l'indice dell'ultimo zero, se esiste
    if 0 in lst:
        last_zero_index = len(lst) - 1 - lst[::-1].index(0)
        lst[last_zero_index] = last_value
    return lst

def compute_downside_upperside_risk(
    anni, fabbisogno, covered, solar,
    anni_prezzi, media_pun, predictive, p95, p5, frwd, budget, observation_period
):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle

    # === Calcolo Open Position ===
    open_position_no_solar = [f - c for f, c in zip(fabbisogno, covered)]
    open_position = [f - (c + s) for f, c, s in zip(fabbisogno, covered, solar)]

    df_open = pd.DataFrame({
        "Anno": anni,
        "Fabbisogno": fabbisogno,
        "Covered": covered,
        "Solar": solar,
        "Open Position (no solar)": open_position_no_solar,
        "Open Position": open_position
    })

    df_prezzi = pd.DataFrame({
        "Year": anni_prezzi,
        "Media PUN": media_pun,
        "Predictive": predictive,
        "95° percentile": p95,
        "5° percentile": p5,
        "frwd": frwd,
        "budget": budget
    })

    # === Calcolo rischio ===
    df_risk = pd.DataFrame({"Year": anni_prezzi})

    def calc_diff(col1, col2):
        return [((a - b) if a is not None and b is not None else None) for a, b in zip(col1, col2)]

    op_map = dict(zip(anni, open_position))
    op_nos_map = dict(zip(anni, open_position_no_solar))

    open_pos = [op_map.get(y, None) * 1000 if op_map.get(y, None) is not None else None for y in anni_prezzi]
    open_pos_nos = [op_nos_map.get(y, None) * 1000 if op_nos_map.get(y, None) is not None else None for y in anni_prezzi]

    def calc_product(diff, op):
        return [(d * o if d is not None and o is not None else None) for d, o in zip(diff, op)]

    diff_95_budget = calc_diff(p95, budget)
    diff_95_frwd = calc_diff(p95, frwd)
    diff_budget_5 = calc_diff(budget, p5)
    diff_frwd_5 = calc_diff(frwd, p5)

    df_risk["Downside Budget"] = calc_product(diff_95_budget, open_pos)
    df_risk["Downside Budget (no solar)"] = calc_product(diff_95_budget, open_pos_nos)
    df_risk["Downside Forward"] = calc_product(diff_95_frwd, open_pos)
    df_risk["Downside Forward (no solar)"] = calc_product(diff_95_frwd, open_pos_nos)
    df_risk["Upside Budget"] = calc_product(diff_budget_5, open_pos)
    df_risk["Upside Budget (no solar)"] = calc_product(diff_budget_5, open_pos_nos)
    df_risk["Upside Forward"] = calc_product(diff_frwd_5, open_pos)
    df_risk["Upside Forward (no solar)"] = calc_product(diff_frwd_5, open_pos_nos)

    # Replace zero with None
    df_open.replace(0, None, inplace=True)
    df_prezzi.replace(0, None, inplace=True)
    df_risk.replace(0, None, inplace=True)

    # === Creazione grafico ===
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.axvspan(2019.5, 2023.5, color="#ffffff", alpha=0.4)
    ax.axvspan(2023.5, 2027.5, color="#ffffff", alpha=0.2)
    anno_oggi = pd.Timestamp.today().year
    ax.axvline(anno_oggi, color="gray", linestyle="--")
    
    ax.text(2020.2, 310, "Historical data", fontsize=12, color="#5e7ab0", backgroundcolor="#daecff")
    ax.text(2024.2, 310, "Predictive data", fontsize=12, color="#5e7ab0", backgroundcolor="#daecff")
    ax.text(anno_oggi, -10, f"{observation_period}", ha="center", fontsize=10, color="gray")

    def replace_last_zero_with_value(lst, last_value):
        if 0 in lst:
            last_zero_index = len(lst) - 1 - lst[::-1].index(0)
            lst[last_zero_index] = last_value
        return lst

    last_historical_value = next((x for x in reversed(media_pun) if x != 0), None)
    p95 = replace_last_zero_with_value(p95, last_historical_value)
    p5 = replace_last_zero_with_value(p5, last_historical_value)
    frwd = replace_last_zero_with_value(frwd, last_historical_value)
    predictive = replace_last_zero_with_value(predictive, last_historical_value)

    def plot_line(data, label, color, annotate=True):
        filtered_data = [d if d != 0 else None for d in data]
        ax.plot(anni_prezzi, filtered_data, label=label, color=color, linewidth=2.5)
        if annotate:
            for x, y in zip(anni_prezzi, filtered_data):
                if y is not None:
                    ax.text(x, y + 5, f"{int(y)}€", ha="center", fontsize=10, color=color)

    plot_line(media_pun, "Media PUN", "#000000")
    if any(val != 0 for val in predictive):
        plot_line(predictive, "market prediction", "#1f77b4")
    plot_line(p95, "95th perc", "#d62728")
    plot_line(p5, "5th perc", "#2ca02c")
    plot_line(frwd, "Forward values", "#FFD700")

    ax.set_title("PUN €/MWh yearly basis", fontsize=16, weight="bold", color="white",
                 bbox=dict(facecolor="#28488d", edgecolor="none", boxstyle="round,pad=0.5"))
    ax.set_xlabel("year", fontsize=12)
    ax.set_ylabel("€ x MWh", fontsize=12)
    ax.set_xlim(2019.5, 2027.5)
    ax.set_ylim(0, 330)
    ax.set_xticks(anni_prezzi)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend()

    plt.tight_layout()

    # === Restituisco tutto in memoria ===
    return df_risk, df_open, df_prezzi, fig


def var_ebitda_risk(periodo_di_analisi, df_risk, font_path='TIMSans-Medium.ttf'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.font_manager as fm
    import numpy as np

    df_risk.dropna(inplace=True)
    anni = df_risk['Year']
    data_sez1_no_solar = np.round(df_risk['Downside Budget (no solar)']/1_000_000, 1)
    data_sez1_solar = np.round(df_risk['Downside Budget']/1_000_000, 1)

    # --- Stile e colori ---
    plt.style.use('seaborn-v0_8-whitegrid')
    colore_sfondo = '#f2f2f2'
    colore_barre = '#c00000'
    colore_testo_barre = '#ffffff'
    colore_titolo = '#28488d'
    colore_etichette_anni = '#000000'
    colore_linea_divisoria = '#dadada'
    colore_testo = '#335193'

    prop = fm.FontProperties(fname=font_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True, gridspec_kw={'height_ratios':[1,1],'hspace':0.25})
    fig.patch.set_facecolor(colore_sfondo)

    # Titolo e periodo di analisi
    fig.text(0.5, 1.02, 'VaR (EBITDA@Risk)', ha='center', va='center', fontsize=14, color=colore_testo,
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'), fontproperties=prop)
    fig.text(0.5, 0.98, periodo_di_analisi, ha='center', va='center', fontsize=14, color='white',
             bbox=dict(facecolor='#404040', edgecolor='none', boxstyle='round,pad=0.5'), fontproperties=prop)

    # Rettangolo bordo
    bordo = Rectangle((0,0),1,1, transform=fig.transFigure, fill=False, edgecolor='black', linewidth=1.5)
    fig.patches.append(bordo)

    larghezza_barra = 0.6
    y_pos = np.arange(len(anni))
    for ax in axes:
        ax.set_facecolor(colore_sfondo)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)

    # Grafico w/o solar
    axes[0].barh(y_pos, data_sez1_no_solar, larghezza_barra, color=colore_barre)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(anni, color=colore_etichette_anni, fontweight='bold', fontsize=12, fontproperties=prop)
    axes[0].set_title('w/o solar', loc='left', color=colore_titolo, fontweight='bold', fontproperties=prop)
    for i, v in enumerate(data_sez1_no_solar):
        axes[0].text(v/2, y_pos[i], f'{v:.1f} mln €', color=colore_testo_barre, va='center', ha='center', fontweight='bold', fontproperties=prop)
        axes[0].plot([0,0],[y_pos[i]-0.4, y_pos[i]+0.4], color='black', linewidth=0.5)
    axes[0].invert_yaxis()

    # Linea divisoria
    line = plt.Line2D([0.05,0.95],[0.525,0.525], transform=fig.transFigure, color=colore_linea_divisoria, linewidth=1)
    fig.add_artist(line)

    # Grafico w/ solar
    axes[1].barh(y_pos, data_sez1_solar, larghezza_barra, color=colore_barre)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(anni, color=colore_etichette_anni, fontweight='bold', fontsize=12, fontproperties=prop)
    axes[1].set_title('w solar', loc='left', color=colore_titolo, fontweight='bold', fontproperties=prop)
    for i, v in enumerate(data_sez1_solar):
        axes[1].text(v/2, y_pos[i], f'{v:.1f} mln €', color=colore_testo_barre, va='center', ha='center', fontweight='bold', fontproperties=prop)
        axes[1].plot([0,0],[y_pos[i]-0.4, y_pos[i]+0.4], color='black', linewidth=0.5)
    axes[1].invert_yaxis()

    plt.tight_layout(rect=[0,0.03,1,0.95])
    return fig  # <-- restituisce la figura direttamente


