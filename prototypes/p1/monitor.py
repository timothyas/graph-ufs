#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    ds = xr.load_dataset("/lustre/stacked-p1-data/loss.nc")

    fig, axs = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)

    axLR = axs[0].twinx()
    ds.loss.plot(ax=axs[0], color="C0")
    ds.learning_rate.plot(ax=axLR, color="C2")

    ds.loss_avg.plot(ax=axs[1], label="Training")
    ds.loss_valid.plot(ax=axs[1], label="Validation")

    for ax in axs:
        for key in ["right", "top"]:
            ax.spines[key].set_visible(False)
    axLR.spines["top"].set_visible(False)

    # labels and stuff
    axs[0].set(
        xlabel="Optimization Step",
        ylabel="Loss Value",
    )
    axLR.set(ylabel="Learning Rate")
    axs[1].set(
        xlabel="Epoch",
        ylabel="Loss Value",
    )
    axs[1].legend()

    fig.savefig("loss.pdf")
