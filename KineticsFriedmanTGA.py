# Code for TGA Friedman Kinetic Analysis
# made by Daniel Martinez Arias, Synthesis Laboratory USFQ
# https://researchusfq.wixsite.com/orgsynth

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dataacq(fname, cextract):
    try:
        data = pd.read_csv(fname)
        columns = [np.array(data.iloc[:, col].tolist()) for col in cextract]
        return columns
    
    except FileNotFoundError:
        print(f"Error: The file '{fname}' has not been found.")
    except Exception as e:
        print(f"Error to process: {e}")
        return []

def dif(x, y):
    if len(x) != len(y):
        raise ValueError("Length!!")
    dy = np.diff(y)
    dx = np.diff(x)
    
    der = dy/dx
    return der

def smooth_algorithm(y, window_size, order, deriv=0, rate=1.0):
  from math import factorial
  try:
    window_size = np.abs(np.int64(window_size))
    order = np.abs(np.int64(order))
  except ValueError:
    raise ValueError("window_size and order have to be of type int")
  if window_size % 2 != 1 or window_size < 3:
    raise ValueError("window_size size must be a positive odd number greater than 2.")
  if order > window_size - 1:
    raise ValueError("order has to be less than window_size - 1.")
  if deriv > order:
    raise ValueError("deriv must be less than order.")

  # precompute coefficients
  half_window = (window_size - 1) // 2
  b = np.asmatrix([[k**i for i in range(order + 1)] for k in range(-half_window, half_window + 1)])
  m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

  # pad the signal at the extremes with
  # values taken from the signal itself
  # (assuming y is a vector now)
  firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
  lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
  y = np.concatenate((firstvals, y, lastvals))
  return np.convolve(m[::-1], y, mode='valid')

def smooth_f(x,y):
    while True:
        # Solicitar el valor de smooth
        smooth_value = np.int64(input("Set the value of smooth (odd number greater than or equal to 5): "))
    
        # Aplicar el filtro de Savitzky-Golay
        y_smooth = smooth_algorithm(y, smooth_value, 3)
    
        # Mostrar la gráfica
        plt.plot(x, y, label='Original data',linewidth=0.5)
        plt.plot(x, y_smooth, label='Smoothed data',linewidth=2)
        plt.legend()
        plt.show()
    
        # Preguntar si se desea continuar
        continuar = input("¿Do you want to set a different value? (y/n): ")
        if continuar.lower() != 'y':
            break
        
    return y_smooth

def ind(data):
    start = 0.1
    end = 0.9
    step = 0.05
    data1 = np.array(data[0])
    
    targets = np.arange(start, end + step, step)
    positions = []
    
    for target in targets:
        index = np.where(np.isclose(data1, target, atol=5e-4))[0]
        positions.append(index)
    index_p = [pos[0] for pos in positions if len(pos) > 0]
    return index_p

def filter(data):
    data = np.array(data)
    indices = np.array(ind(data))
    
    data_k = []
    for index in indices:
        data_k.append(data[:,index])   
          
    return np.array(data_k)

def linear_r(x,y):
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_total = np.sum((y - y_mean)**2)  
    ss_residual = np.sum((y - y_pred)**2)  
    r_squared = 1 - (ss_residual / ss_total)
    
    return slope, intercept, r_squared
# ----------------------------------------------------------------------------
# Kinetic data
# Files ---------------------------EDIT---------------------------------------
b1 = 'c10.csv'
b2 = 'c15.csv'
b3 = 'c20.csv'
b4 = 'c25.csv'

# Heating rates --------------------EDIT--------------------------------------
hr1 = 10
hr2 = 15
hr3 = 20
hr4 = 25
# ----------------------------------------------------------------------------

v = [dataacq(b1, [1,3,8]), dataacq(b2, [1,3,8]), dataacq(b3, [1,3,8]), dataacq(b4, [1,3,8])]
m = []
t = []
h = []
a = []

# Data extraction
for i in range(4): 
    m.append(v[i][0])  # Mass
    t.append(v[i][1])  # Temperature
    h.append(v[i][2])  # Heat Flow
    
    # Conversion \alpha
    a.append((v[i][0][0] - v[i][0]) / (v[i][0][0] - v[i][0][-1]))

# da/dT
da1 = dif(t[0],a[0])
da2 = dif(t[1],a[1])
da3 = dif(t[2],a[2])
da4 = dif(t[3],a[3])

t1 = np.delete(t[0],-1) + 273.15
t2 = np.delete(t[1],-1) + 273.15
t3 = np.delete(t[2],-1) + 273.15
t4 = np.delete(t[3],-1) + 273.15

# Smooth for da/dT
da1_s = smooth_f(t1,da1)
da2_s = smooth_f(t2,da2)
da3_s = smooth_f(t3,da3)
da4_s = smooth_f(t4,da4) #tested----------------------------------------------

# ln(\beta da/dT_s)
lnb1 = np.log(hr1 * da1_s) if da1_s is not None else 0
lnb2 = np.log(hr2 * da2_s) if da2_s is not None else 0
lnb3 = np.log(hr3 * da3_s) if da3_s is not None else 0
lnb4 = np.log(hr4 * da4_s) if da4_s is not None else 0

# Smooth for ln(\beta da/dT_s)
lnb1_s = smooth_f(t1, lnb1)
lnb2_s = smooth_f(t2, lnb2)
lnb3_s = smooth_f(t3, lnb3)
lnb4_s = smooth_f(t4, lnb4)

# Group of data
t1i = 1/t1
t2i = 1/t2
t3i = 1/t3
t4i = 1/t4

a1 = np.delete(a[0],-1)
a2 = np.delete(a[1],-1)
a3 = np.delete(a[2],-1)
a4 = np.delete(a[3],-1)

g1 = [a1, t1i, lnb1_s]
g2 = [a2, t2i, lnb2_s]
g3 = [a3, t3i, lnb3_s]
g4 = [a4, t4i, lnb4_s]

k1 = filter(g1)
k2 = filter(g2)
k3 = filter(g3)
k4 = filter(g4)

k1 = k1[:, 1:]  
k2 = k2[:, 1:]
k3 = k3[:, 1:]
k4 = k4[:, 1:]

kinetics_data = []
for i in range(min(k1.shape[0], k2.shape[0], k3.shape[0], k4.shape[0])):
    combined_row = np.vstack([k1[i], k2[i], k3[i], k4[i]])
    kinetics_data.append(combined_row)

kinetics_data = np.array(kinetics_data)

kinetic_parameters = []
for i in range(kinetics_data.shape[0]):
    temperatures = kinetics_data[i, :, 0] 
    ln_values = kinetics_data[i, :, 1]   
    
    slope, intercept, r_squared = linear_r(temperatures, ln_values)

    kinetic_parameters.append({
        'alpha': 0.05*i + 0.1,          # Alpha conversion
        'Ea': -slope*1.987/1000,          # Activation energy [kcal/mol]
        'ln(A)': intercept,  # Intercept
        'R^2': r_squared         
    })
results_df = pd.DataFrame(kinetic_parameters)
print("\nAll data:")
print(results_df.to_string(index=False, float_format="{:.4f}".format))

with open("resultsfriedman.txt", "w") as f:
    f.write("\nAll data:\n")
    f.write(results_df.to_string(index=False, float_format="{:.4f}".format))

summary = pd.DataFrame({
    'Metric': ['Ea [kcal/mol]', 'ln(A)', 'R^2'],
    'Mean': [results_df['Ea'].mean(), results_df['ln(A)'].mean(), results_df['R^2'].mean()],
    'Std Dev': [results_df['Ea'].std(), results_df['ln(A)'].std(), results_df['R^2'].std()],
    'Min': [results_df['Ea'].min(), results_df['ln(A)'].min(), results_df['R^2'].min()],
    'Max': [results_df['Ea'].max(), results_df['ln(A)'].max(), results_df['R^2'].max()],
})
print("\nKinetic parameters:")
print(summary.to_string(index=False, float_format="{:.4f}".format))

with open("resultsfriedman.txt", "a") as f:  
    f.write("\n\n\n")
    f.write("Kinetic parameters:\n")
    f.write(summary.to_string(index=False, float_format="{:.4f}".format))

# Graphics
# Weight vs Temperature Figure
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.plot(t[0], m[0], label=r"$\beta = 10^\circ C/\text{min}$", color='blue')
plt.plot(t[1], m[1], label=r"$\beta = 15^\circ C/\text{min}$", color='green')
plt.plot(t[2], m[2], label=r"$\beta = 20^\circ C/\text{min}$", color='red')
plt.plot(t[3], m[3], label=r"$\beta = 25^\circ C/\text{min}$", color='purple')
plt.xlabel("Temperature [°C]")
plt.ylabel("Weight [mg]")
plt.title("Weight vs Temperature")
plt.legend()

with open("weightvstemp.txt", "w") as f:
    for i, (temp, mass) in enumerate(zip(t, m)):
        f.write(f"Data for β = {10 + 5*i} °C/min\n")
        f.write("Temperature [°C]    Weight [mg]\n")
        for T, M in zip(temp, mass):
            f.write(f"{T:.2f}\t{M:.4f}\n")
        f.write("\n\n\n\n\n\n")
        
# Derivative vs Temeprature Figure
plt.subplot(2,2,2)
plt.plot(t1-273.15, da1_s, label=r"$\beta = 10^\circ C/\text{min}$", color='blue')
plt.plot(t2-273.15, da2_s, label=r"$\beta = 15^\circ C/\text{min}$", color='green')
plt.plot(t3-273.15, da3_s, label=r"$\beta = 20^\circ C/\text{min}$", color='red')
plt.plot(t4-273.15, da4_s, label=r"$\beta = 25^\circ C/\text{min}$", color='purple')
plt.xlabel("Temperature [°C]")
plt.ylabel("DTG [%/°C]")
plt.title("First derivative vs Temperature")
plt.legend()

with open("dtgvstemp.txt", "w") as f:
    for i, (temp, mass) in enumerate(zip([t1-273.15,t2-273.15,t3-273.15,t4-273.15], [da1_s,da2_s,da3_s,da4_s])):
        f.write(f"Data for β = {10 + 5*i} °C/min\n")
        f.write("Temperature [°C]    DTG [%/°C]\n")
        for T, M in zip(temp, mass):
            f.write(f"{T:.2f}\t{M:.4f}\n")
        f.write("\n\n\n\n\n\n")
        
# 0.1 < \alpha < 0.45 Figure
plt.subplot(2, 2, 3) 
for i in range(8):  
    temperatures = kinetics_data[i, :, 0]
    ln_values = kinetics_data[i, :, 1]    
    plt.plot(temperatures, ln_values, label=rf"$\alpha$ = {0.05*i + 0.1:.2f}")
plt.title("Friedman Model Kinetics for 0.1 < $\\alpha$ < 0.45 values")
plt.xlabel("1/T [1/K]")
plt.ylabel(r"$\ln(\beta \frac{d\alpha}{dT})$")
plt.legend(loc="best", fontsize="small", ncol=2)

# 0.5 < \alpha < 0.1
plt.subplot(2, 2, 4)  
for i in range(8, 17):
    temperatures = kinetics_data[i, :, 0]  
    ln_values = kinetics_data[i, :, 1]
    plt.plot(temperatures, ln_values, label=rf"$\alpha$ = {0.05*i + 0.1:.2f}")
plt.title("Friedman Model Kinetics for 0.5 < $\\alpha$ < 0.9 values")
plt.xlabel("1/T [1/K]")
plt.ylabel(r"$\ln(\beta \frac{d\alpha}{dT})$")
plt.legend(loc="best", fontsize="small", ncol=2)

with open("data_friedman.txt", "w") as f:
    f.write("Friedman Model Kinetics (0.1 < alpha < 0.45)\n")
    f.write("alpha\t1/T [1/K]\tln(beta * dα/dT)\n")
    for i in range(8):
        alpha = 0.05 * i + 0.1
        temperatures = kinetics_data[i, :, 0]
        ln_values = kinetics_data[i, :, 1]
        for T, ln_val in zip(temperatures, ln_values):
            f.write(f"{alpha:.2f}\t{T:.6f}\t{ln_val:.6f}\n")
        f.write("\n")
        
    f.write("\nFriedman Model Kinetics (0.5 < alpha < 0.9)\n")
    f.write("alpha\t1/T [1/K]\tln(beta * dα/dT)\n")
    for i in range(8,17): 
        alpha = 0.05 * i + 0.1
        temperatures = kinetics_data[i, :, 0]
        ln_values = kinetics_data[i, :, 1]
        for T, ln_val in zip(temperatures, ln_values):
            f.write(f"{alpha:.2f}\t{T:.6f}\t{ln_val:.6f}\n")
        f.write("\n")

plt.tight_layout()
plt.savefig("kinetics.jpg", dpi=300)
plt.show()

