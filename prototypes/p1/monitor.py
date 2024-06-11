#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    ds = xr.load_dataset("/lustre/stacked-p1-data/loss.nc")

    fig, axs = plt.subplots(1,2, figsize=(4,8))
    ds.loss.plot(ax=axs[0])


    ds.loss_avg.plot(ax=axs[1])
    ds.loss_valid.plot(ax=axs[1])

    for ax in axs:
        for key in ["right", "top"]:
            ax.spines[key].set_visible(False)

    fig.savefig("loss.pdf")
