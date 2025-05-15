import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import tensorflow as tf

hours = np.array([
    8.0, 9.5, 12.0, 15.0, 18.0, 21.0,
    23.5, 0.0, 1.5, 3.0, 5.0, 6.5
])
durations = np.array([
    35, 40, 25, 20, 45, 50,
    60, 30, 28, 22, 18, 20
])

X_nn = (hours / 24.0).reshape(-1, 1)
X_poly = hours.reshape(-1, 1)
y = durations

model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_nn.compile(optimizer='adam', loss='mse')
model_nn.fit(X_nn, y, epochs=500, verbose=0)

degree = 3
model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_poly.fit(X_poly, y)

def time_to_float(hour, minute):
    return hour + minute / 60

times = [(10, 30), (0, 0), (2, 40)]
times_float = [time_to_float(h, m) for h, m in times]
inputs_nn = np.array(times_float) / 24.0
inputs_poly = np.array(times_float).reshape(-1, 1)

preds_nn = model_nn.predict(inputs_nn)
preds_poly = model_poly.predict(inputs_poly)

print("=== ПРОГНОЗИ ===")
for i, (h, m) in enumerate(times):
    print(f"\n⏰ {h:02}:{m:02}")
    print(f"   🧠 Нейронна мережа: {preds_nn[i][0]:.2f} хв")
    print(f"   📈 Поліноміальна регресія: {preds_poly[i]:.2f} хв")

x_plot = np.linspace(0, 24, 200)
x_plot_nn = x_plot / 24.0
y_plot_nn = model_nn.predict(x_plot_nn)
y_plot_poly = model_poly.predict(x_plot.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.scatter(hours, y, color='black', label='Тренувальні дані')
plt.plot(x_plot, y_plot_nn, 'r-', label='Нейронна мережа')
plt.plot(x_plot, y_plot_poly, 'b--', label=f'Поліноміальна регресія (ступінь {degree})')
plt.title("Прогноз тривалості поїздки за часом доби")
plt.xlabel("Години")
plt.ylabel("Тривалість поїздки (хв)")
plt.grid(True)
plt.legend()
plt.show()
