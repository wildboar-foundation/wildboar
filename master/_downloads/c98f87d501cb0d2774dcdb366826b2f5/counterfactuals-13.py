def plot_counterfactual(i, X_test, y_test, y_pred):
   plt.plot(
      X_test[i],
      label="original (y_pred = %d, y_actual = %d)" % (y_pred[i], y_test[i]),
      lw=0.5,
   )
   plt.plot(X_cf[i], label="counterfactual (y = %d)" % cf_pred[i], lw=0.5)
   plt.plot(
      np.mean(X_test[y_test == cf_pred[i]], axis=0),
      linestyle="dashed",
      label="mean of X with y = %d" % cf_pred[i],
      lw=0.5,
   )
   plt.legend()
   plt.title("Sample #%d" % i)

plt.figure()
plot_counterfactual(4, X_test, y_test, y_pred)
plt.figure()
plot_counterfactual(15, X_test, y_test, y_pred)
plt.figure()
plot_counterfactual(36, X_test, y_test, y_pred)