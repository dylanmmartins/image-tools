# Image tools

## Installation

In a terminal, change to the directory where this repository is cloned and run `pip install -e .` and the importable package `imagetools` will be installed into the current conda environment.

### Writing matplotlib animations

Now in your script you can import it (here I'm using the alias `imgt`) alongside the other packes we'll use in the demo.

```
import numpy as np
import matplotlib.pyplot as plt
import imagetools as imgt
```

The animation needs a custom function that makes your matplotlib figure. Anything that varies in each frame of your desired video needs to happen in the arguments to this function. In this example, `shift` is a `float` value that shifts a sine functin up and down. Each frame of our animatin will use a differnet `shift` value.

At the end of the plotting function once the plotting is done, we include the line `img = imgt.fmt_figure(fig1)` the matplotlib figure object `fig1` (or whatever you named yours) will be convwerted to an array of pixel values in this case named `img`. This works for any figure object, so it can be a single-subplot figure, gridspec, etc. This function needs to return the img object.

```
def my_plotting_function(shift):

    fig1, ax = plt.subplot(1, 1, figsize=(4,3), dpi=300)

    # Plot whatever you'd like and set properties
    ax.plot(range(20), np.sin(range(20))+shift)
    ax.set_xlim([0, 20])
    ax.set_ylim([-5, 5])
    fig1.tight_layout()

    # Convert the figure object to an array-like image
    img = imgt.fmt_figure(fig1)

    return img
```

Any number of arguments is okay for this function, but each needs to be array-like where the 0th dimension of all arguments have a shared size. That shared 0th dimension size will also be the number of frames in the animated video. In this case, `shift` will be an array of shape `(30,)`.
Each frame in the demo video will shift a sine function by one these values
shift_values = np.linspace(-4, 4, 30)

frame_stack = imgt.make_frames(my_plotting_function, [shift_values])

imgt.write_animation(
    frame_stack,
    fps=20,
    savepath='T:/shifted_sine_animation.mp4'
)
