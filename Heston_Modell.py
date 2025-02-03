###########################################
##### Kalibrierung des Heston Modells #####
###########################################

import numpy as np
from scipy.optimize import least_squares # Zur Minimierung der Fehlerfunktion
from scipy.special import roots_legendre # Gauß-Quadratur zur Berechnung der Integrale für P1 und P2 aus dem Optionspreis
import matplotlib.pyplot as plt # Zum Plotten der Ergebnisse
import yfinance as yf # Zum Importieren von echten Marktpreisen 
from scipy.stats import norm # Normen für Black-Scholes
from datetime import datetime # Zur Bestimmung der Laufzeit 

n = 100 # Zur Bestimmung der quadrature order bzgl. der Berechnung der Gauß-Quadratur

# theta = [kappa, sigma, rho, v0, vmean] # Parameter theta, den wir kalibrieren wollen

# Charakteristische Funktion des Heston Modells 
def characteristic_function(S0, r, u, t, theta):

   # Parameter definieren 
   [kappa, sigma, rho, v0, vmean] = theta 

   # Definieren der Parameter xi, d, A und D analog zu (3.7) und (3.8) für die charakteristische Funktion 
   xi = kappa - sigma * rho * 1j * u
   d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
   
   sinh = np.sinh(d * t / 2)
   cosh = np.cosh(d * t / 2)
   
   A1 = (u**2 + 1j * u) * sinh
   A2 = d * cosh + xi * sinh 
   A2 = np.where(np.abs(A2) < 1e-10, 1e-10, A2) # Um Probleme mit zu kleinen Werten zu vermeiden
   A = A1 / A2
   
   D = np.log(np.maximum(1e-10, d)) + kappa * t / 2 - np.log(A2)  # Verhindert log(0)
   
   # Definieren der charakteristischen Funktion analog zu (3.8)
   phi = np.exp( 1j * u * (np.log(S0) + r * t) - v0 * A + 2 * kappa * vmean / sigma**2 * D)

   return phi

# Preis einer europäischen Call-Option im Heston Modell
def option_price(theta, K, T, r, S0):

   # Parameter definieren 
   [kappa, sigma, rho, v0, vmean] = theta 

   # Gauß-Quadratur
   x, w = roots_legendre(n) # Gauß-Knoten und Gewichte für [-1, 1]
   # Da die Integrale für P1 und P2 von 0 bis unendlich laufen, nehmen wir noch eine Reskalierung vor 
   x = 1/2 * (x+1) * 400 # Reskalierung zu [0, 400]
   w = 1/2 * 400 * w # Reskalierung der Gewichte 
   
   # Definieren von P1 und P2 
   # a und b dienen hier zur übersichtlicheren Darstellung von P1 und P2 
   a = np.exp(-1j * x* np.log(K))
   b = 1j * x

   # Summation über die Gauß-Knotenpunkte und Gewichte zum Integrieren von P1 und P2 
   P1 = 0.5 + (1 / np.pi) * np.sum(
       w * np.real(a / (b * S0 * np.exp(r * T)) * characteristic_function(S0, r, x - 1j, T, theta))
   )
   P2 = 0.5 + (1 / np.pi) * np.sum(
       w * np.real(a / b * characteristic_function(S0, r, x, T, theta))
   )

   # Optionspreis in dem Heston Modell
   return S0 * P1 - np.exp(-r*T) * K * P2 

# Jacobi-Matrix des Optionswertes
def jacobi_heston(theta, strikes, maturities, market_prices, S0, r):
    
    # Parameter definieren
    [kappa, sigma, rho, v0, vmean] = theta

    # Initialisierung der Jacobi-Matrix
    jacobi_matrix = np.zeros((len(strikes), len(theta)))

    # Gauß-Quadratur Vorbereitung
    x, w = roots_legendre(n)
    x = 1 / 2 * (x + 1) * 400  # Reskalierung auf [0, 400]
    w = 1 / 2 * 400 * w

    for i, (K, T) in enumerate(zip(strikes, maturities)):
        
        # Ableitungen der charakteristischen Funktion
        # Die Ableitungen werden analog zu Cui et al. (3.3 Analytical Gradient) definiert
        def characteristic_function_derivatives(u):
            xi = kappa - sigma * rho * 1j * u
            d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
            sinh = np.sinh(d * T / 2)
            cosh = np.cosh(d * T / 2)
            A = (u**2 + 1j * u) * sinh / (d * cosh + xi * sinh)
            D = np.log(np.maximum(1e-10, d)) + kappa * T / 2 - np.log(d * cosh + xi * sinh)

            # Ableitungen von d nach den Parametern
            d_dkappa = (1 - rho * sigma * 1j * u / d) * T / 2
            d_dsigma = ((sigma * (u**2 + 1j * u) / d) + rho * 1j * u * kappa / d) * T / 2
            d_drho = -sigma * 1j * u * T / (2 * d)

            # Ableitungen von A nach den Parametern
            A_dkappa = -A * (d_dkappa / d + T / 2)
            A_dsigma = -A * (d_dsigma / d + T / 2)
            A_drho = -A * d_drho / d

            # Ableitungen von D nach den Parametern
            D_dkappa = T / 2 + 1 / d * d_dkappa
            D_dsigma = 1 / d * d_dsigma
            D_drho = 1 / d * d_drho

            return {
                "A_dkappa": A_dkappa,
                "A_dsigma": A_dsigma,
                "A_drho": A_drho,
                "D_dkappa": D_dkappa,
                "D_dsigma": D_dsigma,
                "D_drho": D_drho
            }
            
        # Ableitungen von P1 und P2 nach den Parametern
        derivs = characteristic_function_derivatives(x)
        P1_dkappa = np.sum(w * np.real(derivs["A_dkappa"]))
        P1_dsigma = np.sum(w * np.real(derivs["A_dsigma"]))
        P1_drho = np.sum(w * np.real(derivs["A_drho"]))

        # Jacobi-Elemente für jeden Parameter berechnen
        jacobi_matrix[i, 0] = P1_dkappa  # Ableitung nach kappa
        jacobi_matrix[i, 1] = P1_dsigma  # Ableitung nach sigma
        jacobi_matrix[i, 2] = P1_drho    # Ableitung nach rho
        jacobi_matrix[i, 3] = 0          # Platzhalter für v0
        jacobi_matrix[i, 4] = 0          # Platzhalter für vmean

    return jacobi_matrix
   
   
# Fehlerfunktion - Differenz zwischen den Modell- und Marktpreisen
def heston_optimize(theta, strikes, maturities, market_prices, S0, r):
   errors = []
   for i in range(len(strikes)):
      model_price = option_price(theta, strikes[i], maturities[i], r, S0) # Modellpreise im Heston-Modell 
      weight = 1 / market_prices[i]  # Gewichtung nach Marktpreis / Kann abweichend auch auf 1 gesetzt werden 
      errors.append(weight * (model_price - market_prices[i])) # Fehler der Modellpreise 
   return np.array(errors)
   
# Black-Scholes für die europäische Call-Option   
def black_scholes(S0, K, T, r, v0):

    # Berechnung von d1 und d2
    d1 = (np.log(S0 / K) + (r + 0.5 * v0**2) * T) / (v0 * np.sqrt(T))
    d2 = d1 - v0 * np.sqrt(T)

    option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return option_price
    

####################################################################
######################## Anfangswerte ##############################
####################################################################

# Risikoloser Zinssatz
r = 0.008 # Zins für S&P 500
#r = 0.12 # Zins für Tesla

# Aktie, auf die wir die Modelle anwenden wollen 
#ticker = yf.Ticker("AAPL") # "AAPL" für Apple
ticker = yf.Ticker("SPY") # "SPY" für S&P 500
#ticker = yf.Ticker("TSLA") # "TSLA" für die Tesla-Aktie 

# Wahl der Laufzeit 
expiry_dates = ticker.options # Alle verfügbaren Laufzeitenden
print("Verfügbare Verfallstermine:", expiry_dates)
# Gewählte Laufzeiten für Tesla 
#chosen_expiry = expiry_dates[4] # Laufzeit ca. 1 Monat 
#chosen_expiry = expiry_dates[8] # Laufzeit ca. 3 Monate 
#chosen_expiry = expiry_dates[15] # Laufzeit ca. 1 Jahr 
# Gewählte Laufzeiten für S&P 500
chosen_expiry = expiry_dates[11] # Laufzeit ca. 1 Monat 
#chosen_expiry = expiry_dates[16] # Laufzeit ca. 3 Monate 
#chosen_expiry = expiry_dates[27] # Laufzeit ca. 1 Jahr 

option_chain = ticker.option_chain(chosen_expiry)

calls = option_chain.calls # Calls mit der gewählten Laufzeit

strikes = calls['strike'].values  # Strike-Preise
market_prices = calls['lastPrice'].values  # Letzter Marktpreis
heute = datetime.now() # Heutiges Datum 
maturity = datetime.strptime(chosen_expiry,"%Y-%m-%d") # Sicherstellen, dass die Formatierung der Daten richtig ist
maturity = (maturity- heute).days /365 # Bestimmen der Laufzeit durch Differenz zwischen gewähltem Laufzeitende und Heute
maturities = [maturity] * len(strikes) # Laufzeit

# Anfangswerte für den Parameter theta 
# [kappa, sigma, rho, v0, vmean]
initial_params = [0.5, 0.275, -0.5, 0.0295, 0.0395] # S&P 500
#initial_params = [0.495, 0.4, -0.4, 0.385, 0.412] # Tesla

S0 = ticker.history(period="1d")['Close'].iloc[-1] # Letzter Schlusskurs der Aktie

####################################################################
###################### Kalibrierung ################################
####################################################################

# Optimierungsproblem inf || r(theta) ||^2 zur Minimierung der Residuen
# Abweichend wird hier der Levenberg-Marquardt Algorithmus verwendet 
# LMA interpoliert zwischen Gauss-Newton und Gradient Descent
# Mit Analytischem Gradienten 
result = least_squares(heston_optimize, initial_params, args=(strikes, maturities, market_prices, S0, r), method="lm", jac=jacobi_heston)
# Ohne Analytischem Gradienten 
#result = least_squares(heston_optimize, initial_params, args=(strikes, maturities, market_prices, S0, r), method="lm")

calibrated_params = result.x # Kalibrierte Parameter 

# Berechnen der Modellpreise für Heston und Black-Scholes
model_prices = [option_price(calibrated_params, strikes[i], maturities[i], r, S0) for i in range(len(strikes))]
# bs_prices = [black_scholes(S0, strikes[i], maturities[i], r, initial_params[3]) for i in range(len(strikes))] # Ohne kalibrierter Volatilität
bs_prices = [black_scholes(S0, strikes[i], maturities[i], r, calibrated_params[3]) for i in range(len(strikes))] # Mit kalibrierter Volatilität

####################################################################
###################### Plotten der Ergebnisse ######################
####################################################################

# Fehlerbestimmung für Heston und Black-Scholes
error = 0
bs_error = 0
for i in range(len(strikes)):
   error = error + abs(model_prices[i] - market_prices[i])
   bs_error = bs_error + abs(bs_prices[i] - market_prices[i])
   
average = sum(market_prices)/len(market_prices) # Durchschnittlicher Preis 

# Printen der verschiedenen Fehler 
print("Marktpreisdurchschnitt: ", average)      
print ("Absoluter Fehler (Heston): ", error)
print ("Absoluter Fehler (Black-Scholes): ", bs_error)
print ("Durchschnittlicher absoluter Fehler (Heston): ", error/len(market_prices))
print ("Durchschnittlicher absoluter Fehler (Black-Scholes): ", bs_error/len(market_prices))
print ("Durchschnittlicher Fehler (Heston): ", (error/len(market_prices))/average * 100)
print ("Durchschnittlicher Fehler (Black-Scholes): ", (bs_error/len(market_prices))/average * 100)

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(strikes, market_prices, 'o-', label='Marktpreise', color='blue')  # Marktpreise-Linie
plt.plot(strikes, bs_prices, 's-', label='Black-Scholes-Preise', color='red') # Black-Scholes-Linie 

# Achsen und Titel
plt.xlabel('Strikes')
plt.ylabel('Optionen-Preise')
plt.title('Vergleich von Markt- und Heston-Modellpreisen')
plt.legend()
plt.grid(True)
#plt.show()

plt.figure(figsize=(8,6))
plt.scatter(market_prices, model_prices, color='blue', label='Heston-Preise')
plt.scatter(market_prices, bs_prices, color='green', label='Black-Scholes-Preise')
plt.plot([min(market_prices), max(market_prices)], [min(market_prices), max(market_prices)], 
         color='red', linestyle='--', label='Ideale Linie')
plt.xlabel('Marktpreise')
plt.ylabel('Modellpreise')
plt.title('Vergleich der Marktpreise und der Modellpreise')
plt.legend()
plt.grid()
#plt.show()

plt.figure(figsize=(8,6))
plt.scatter(market_prices, model_prices, color='blue', label='Heston-Preise')
plt.plot([min(market_prices), max(market_prices)], [min(market_prices), max(market_prices)], 
         color='red', linestyle='--', label='Ideale Linie')
plt.xlabel('Marktpreise')
plt.ylabel('Modellpreise')
plt.title('Vergleich der Marktpreise und der Modellpreise')
plt.legend()
plt.grid()
#plt.show()

plt.figure(figsize=(8,6))
plt.scatter(market_prices, model_prices, color='blue', label='Heston-Preise')
plt.scatter(market_prices, bs_prices, color='green', label='Black-Scholes-Preise')
plt.plot([min(market_prices), max(market_prices)], [min(market_prices), max(market_prices)], 
         color='red', linestyle='--', label='Ideale Linie')
plt.axis([20, 60, 20, 60])
plt.xlabel('Marktpreise')
plt.ylabel('Modellpreise')
plt.title('Vergleich der Marktpreise und der Modellpreise')
plt.legend()
plt.grid()
#plt.show()

plt.figure(figsize=(8,6))
plt.scatter(market_prices, model_prices, color='blue', label='Heston-Preise')
plt.plot([min(market_prices), max(market_prices)], [min(market_prices), max(market_prices)], 
         color='red', linestyle='--', label='Ideale Linie')
plt.axis([20, 60, 20, 60])
plt.xlabel('Marktpreise')
plt.ylabel('Modellpreise')
plt.title('Vergleich der Marktpreise und der Modellpreise')
plt.legend()
plt.grid()
#plt.show()

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(strikes, market_prices, 'o-', label='Marktpreise', color='blue')  # Marktpreise-Linie
plt.plot(strikes, model_prices, 's-', label='Heston-Preise', color='green')  # Modellpreise-Linie
plt.plot(strikes, bs_prices, 's-', label='Black-Scholes-Preise', color='red') # Black-Scholes-Linie 

# Achsen und Titel
plt.xlabel('Strikes')
plt.ylabel('Optionen-Preise')
plt.title('Vergleich von Markt- und Heston-Modellpreisen')
plt.legend()
plt.grid(True)
#plt.show()

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(strikes, market_prices, 'o-', label='Marktpreise', color='blue')  # Marktpreise-Linie
plt.plot(strikes, model_prices, 's-', label='Heston-Preise', color='green')  # Modellpreise-Linie

# Achsen und Titel
plt.xlabel('Strikes')
plt.ylabel('Optionen-Preise')
plt.title('Vergleich von Markt- und Heston-Modellpreisen')
plt.legend()
plt.grid(True)
plt.show()
