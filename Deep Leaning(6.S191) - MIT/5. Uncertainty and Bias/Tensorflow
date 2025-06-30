# Estimating Epistemic Uncertainty - Ensemble
num_ensembles = 5

for i in range(num_ensembles):
  model = create_model(...)
  model.fit(...)

raw_preditions = [models[i].predict(x) for i in range(num_ensembles)]

mu = np.mean(raw_predictions)
uncertainty = np.var(raw_predictions)

# Estimating Epistemic Uncertainty - Dropout
for _in range(T):
  forward_passes.append(model(x, dropout=True))

mu = np.mean(forward_passes)
uncertainty = np.var(forward_passes)
  
