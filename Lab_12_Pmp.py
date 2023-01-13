import arviz as az
import matplotlib.pyplot as plt


az.load_arviz_data("centered_eight")
az.load_arviz_data("non_centered_eight")

#1
idata_centered = az.load_arviz_data("centered_eight")
post_centered = idata_centered.posterior

idata_not_centered = az.load_arviz_data("non_centered_eight")
post_not_centered = idata_not_centered.posterior

print(post_centered)
print(post_not_centered)


#2
rhat1 = az.rhat(idata_centered, var_names=["mu", "theta"])
rhat2 = az.rhat(idata_not_centered, var_names=["mu", "theta"])

print(rhat1)
print(rhat2)

#3
idata_centered.sample_stats.diverging.sum()
idata_not_centered.samplestats.diverging.sum()

ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5), constrained_layout=True)

for idx, tr in enumerate([idata_centered, idata_not_centered]):
    az.plot_pair(tr, var_names=['mu', 'tau'], kind='scatter',
                 divergences=True, divergences_kwargs={'color':'C1'},
                 ax=ax[idx])

    ax[idx].set_title(['centered', 'non-centered'][idx])