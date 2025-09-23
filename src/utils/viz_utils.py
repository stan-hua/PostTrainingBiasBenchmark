"""
viz_utils.py

Description: Used to visualize data
"""

# Standard libraries
import logging
import os

# Non-standard libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib import rc
from matplotlib.container import ErrorbarContainer


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Random seed
SEED = 42

# Color palette
COLORS = [
    "#4C9AC9",  # Sky Blue
    "#F1A7C1",  # Soft Pink
    "#F2B87D",  # Golden Yellow
    "#B85C5C",  # Warm Red
    "#C6D89E",  # Light Olive Green
    "#85B8A2",  # Soft Teal
    "#7D85A7",  # Slate Blue
    "#F4A2A2",  # Light Coral Red
    "#D3A9F5",  # Lavender
    "#A9D9B2",  # Pale Green
    "#FF7F4C",  # Bright Orange
    "#6E82B5",  # Dusty Blue
    "#C78E72",  # Muted Clay
    "#F6D07F",  # Warm Yellow
]

# Create diverging palette
DIVERGING_PALETTE = sns.diverging_palette(h_neg=120, h_pos=10, s=50, l=70, sep=1, n=6, center='light')

PALETTE = {
    "ERM": "#60B2E5", "RWY": "#F9F5E3", "RWG": "#E95C47",
    "RWY-ES": "#E9C5C5", "RWG-ES": "#E97C7C", "GDRO-ES": "#1F497D"
}


################################################################################
#                              Plotting Functions                              #
################################################################################
def set_theme(tick_scale=1.3, figsize=(10, 6), style="whitegrid", rcparams=None):
    """
    Create scientific theme for plot
    """
    custom_params = {
        "axes.spines.right": False, "axes.spines.top": False,
        "figure.figsize": figsize,
    }
    if rcparams is not None:
        custom_params.update(rcparams)
    sns.set_theme(style=style, font_scale=tick_scale, rc=custom_params)


def catplot(
        df, x=None, y=None, hue=None,
        bar_labels=None, exclude_bar_labels=False,
        palette=COLORS,
        plot_type="bar",
        vertical_lines=None,
        figsize=None,
        title=None, title_size=None,
        xlabel=None,
        ylabel=None,
        x_lim=None,
        y_lim=None,
        tick_params=None,
        legend=False,
        horizontal_legend=False,
        save_dir=None,
        save_fname=None,
        **extra_plot_kwargs,
    ):
    """
    Creates a categorical plot. One of bar/count/pie

    Note
    ----
    For bar plots, has option to add label on bar plot

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data for the bar plot.
    x : str
        Column name for the x-axis variable.
    y : str
        Column name for the y-axis variable.
    hue : str, optional
        Column name for grouping variable that will produce bars with different colors, by default None.
    bar_labels : list, optional
        List of text to place on top of each bar, by default None.
    exclude_bar_labels : bool, optional
        If True, exclude bar labels, by default None.
    palette : list, optional
        List of colors to use for the bars, by default None.
    plot_type : str, optional
        Type of plot to create, by default "bar".
    vertical_lines : list, optional
        List of x-axis values to draw vertical lines at, by default None.
    figsize : tuple, optional
        Tuple specifying the figure size, by default None.
    title : str, optional
        Title for the plot, by default None.
    title_size : int, optional
        Font size for title, by default 17
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    x_lim : tuple, optional
        Tuple specifying the x-axis limits, by default None.
    y_lim : tuple, optional
        Tuple specifying the y-axis limits, by default None.
    tick_params : dict, optional
        Dictionary specifying the tick parameters, by default None
    legend : bool, optional
        Whether to include a legend, by default False.
    save_dir : str, optional
        Directory to save the plot, by default None.
    save_fname : str, optional
        Filename to save the plot, by default None.
    extra_plot_kwargs : dict, optional
        Additional keyword arguments to pass to the plot function, by default {}

    Returns
    -------
    matplotlib.axes.Axes
        Returns the Axes object with the plot for further tweaking.
    """
    # NOTE: If palette and colors provided, always chose color
    if palette and (extra_plot_kwargs.get("color") or extra_plot_kwargs.get("colors")):
        palette = None

    # Create plot keyword arguments
    plot_kwargs = {
        "data": df, "x": x, "y": y, "hue": hue,
        "legend": legend,
        "palette": palette,
        **extra_plot_kwargs,
    }
    # Add colors, if not seaborn function
    if plot_type in ["bar_with_ci", "grouped_bar_with_ci", "pie"]:
        color_key = "colors" if plot_type == "pie" else "color"
        if color_key not in plot_kwargs:
            plot_kwargs[color_key] = sns.color_palette()
            if palette:
                plot_kwargs[color_key] = sns.color_palette(palette)
        plot_kwargs.pop("palette", None)

    # Raise error, if plot type is invalid
    supported_types = ["bar", "bar_with_ci", "grouped_bar_with_ci", "count", "pie", "hist", "kde", "heatmap"]
    if plot_type not in supported_types:
        raise ValueError(f"Invalid plot type: `{plot_type}`! See supported types: {supported_types}")

    # Create plot based on requested function
    plot_func = None
    if plot_type == "bar":
        plot_func = sns.barplot
        add_default_dict_vals(plot_kwargs, width=0.95)
    elif plot_type == "bar_with_ci":
        plot_func = barplot_with_ci
        add_default_dict_vals(plot_kwargs, width=0.95)
        remove_dict_keys(plot_kwargs, ["hue", "legend"])
    elif plot_type == "grouped_bar_with_ci":
        plot_func = grouped_barplot_with_ci
        remove_dict_keys(plot_kwargs, ["legend"])
    elif plot_type == "count":
        plot_func = sns.countplot
        add_default_dict_vals(plot_kwargs, width=0.95)
    elif plot_type == "hist":
        plot_func = sns.histplot
    elif plot_type == "kde":
        plot_func = sns.kdeplot
    elif plot_type == "pie":
        assert x, "Must specify x-axis variable for pie plot"

        # Remove incompatible keys
        remove_dict_keys(plot_kwargs, ["data", "y", "hue", "legend", "palette"])
        # Add defaults
        add_default_dict_vals(
            plot_kwargs, autopct="%1.1f%%", startangle=140,
            radius=0.5, shadow=True
        )

        # Count the occurrences of each category
        counts = df[x].value_counts()
        plot_kwargs["x"] = counts
        plot_kwargs["labels"] = counts.index

        # Create the pie chart
        plot_func = plt.pie
    elif plot_type == "heatmap":
        plot_func = heatmap
        remove_dict_keys(plot_kwargs, ["x", "y", "hue", "palette", "legend"])

    # Create figure
    if figsize is not None:
        plt.figure(figsize=figsize)

    # Create plot
    ax = plot_func(**plot_kwargs)

    # Add bar labels
    if not exclude_bar_labels and (bar_labels or plot_type == "count"):
        bar_kwargs = {"labels": bar_labels}
        if plot_type == "count":
            bar_kwargs.pop("labels")
        bar_label_fmt = bar_kwargs.get("fmt", "%.1f")

        # Add bar labels
        for container in ax.containers:
            if container is None:
                print("Bar encountered that is empty! Can't place label...")
                continue
            if isinstance(container, ErrorbarContainer):
                print("Skipping labeling of error bar container...")
                continue
            ax.bar_label(container, fmt=bar_label_fmt, size=12, weight="semibold", **bar_kwargs)

    # If specified, add vertical lines
    if vertical_lines:
        for curr_x in vertical_lines:
            ax.axvline(x=curr_x, color="black", linestyle="dashed", alpha=0.5)

    # Perform post-plot logic
    ax = post_plot_logic(
        ax=ax,
        title=title, title_size=title_size,
        xlabel=xlabel, ylabel=ylabel,
        x_lim=x_lim, y_lim=y_lim,
        tick_params=tick_params,
        legend=legend and plot_type not in ["hist", "kde", "count", "bar"],
        horizontal_legend=horizontal_legend,
        save_dir=save_dir, save_fname=save_fname,
    )
    return ax


def numplot(
        df, x=None, y=None, hue=None,
        palette=COLORS,
        plot_type="box",
        vertical_lines=None,
        figsize=None,
        title=None, title_size=None,
        xlabel=None,
        ylabel=None,
        x_lim=None,
        y_lim=None,
        tick_params=None,
        legend=False,
        horizontal_legend=False,
        save_dir=None,
        save_fname=None,
        violin_kwargs=None,
        **extra_plot_kwargs,
    ):
    """
    Creates a numeric plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data for the bar plot.
    x : str
        Column name for the x-axis variable.
    y : str
        Column name for the y-axis variable.
    hue : str, optional
        Column name for grouping variable that will produce bars with different colors, by default None.
    palette : list, optional
        List of colors to use for the bars, by default None.
    plot_type : str, optional
        Type of plot to create, by default "box".
    vertical_lines : list, optional
        List of x-axis values to draw vertical lines at, by default None.
    figsize : tuple, optional
        Tuple specifying the figure size, by default None.
    title : str, optional
        Title for the plot, by default None.
    title_size : int, optional
        Font size for title, by default 17
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    x_lim : tuple, optional
        Tuple specifying the x-axis limits, by default None.
    y_lim : tuple, optional
        Tuple specifying the y-axis limits, by default None.
    tick_params : dict, optional
        Keyword arguments to pass into `matplotlib.pyplot.tick_params`, by default None
    legend : bool, optional
        Whether to include a legend, by default False.
    save_dir : str, optional
        Directory to save the plot, by default None.
    save_fname : str, optional
        Filename to save the plot, by default None.
    violin_kwargs : dict, optional
        Violin plot specific keyword arguments
    extra_plot_kwargs : dict, optional
        Additional keyword arguments to pass to the plot function, by default {}

    Returns
    -------
    matplotlib.axes.Axes
        Returns the Axes object with the plot for further tweaking.
    """
    # Create plot keyword arguments
    plot_kwargs = {
        "data": df, "x": x, "y": y, "hue": hue,
        "legend": legend,
        "palette": palette,
        **extra_plot_kwargs,
    }

    # Raise error, if plot type is invalid
    supported_types = ["box", "strip", "line", "scatter", "violin", "displot"]
    if plot_type not in supported_types:
        raise ValueError(f"Invalid plot type: `{plot_type}`! See supported types: {supported_types}")

    # Create plot based on requested function
    plot_func = None
    if plot_type == "box":
        plot_func = sns.boxplot
        default_kwargs = {"width": 0.95}
        plot_kwargs.update({k: v for k, v in default_kwargs.items() if k not in plot_kwargs})
    elif plot_type == "strip":
        plot_func = sns.stripplot
    elif plot_type == "line":
        plot_func = sns.lineplot
    elif plot_type == "scatter":
        plot_func = sns.scatterplot
    elif plot_type == "violin":
        plot_func = sns.violinplot
    elif plot_type == "displot":
        plot_func = sns.displot

    # Create figure
    if figsize is not None:
        plt.figure(figsize=figsize)

    # Create plot
    ax = plot_func(**plot_kwargs)

    # If specified, add violin specific keyword arguments
    if violin_kwargs:
        if violin_kwargs.get("split_violin"):
            arg_names = ["threshold", "flip_idx", "colors"]
            split_kwargs = {k:v for k,v in violin_kwargs.items() if k in arg_names}
            split_violin_by_threshold(ax, **split_kwargs)

    # If specified, add vertical lines
    if vertical_lines:
        for curr_x in vertical_lines:
            ax.axvline(x=curr_x, color="black", linestyle="dashed", alpha=0.5)

    # Perform post-plot logic
    # CASE 1: FacetGrid
    if isinstance(ax, sns.axisgrid.FacetGrid):
        axes = ax.axes.flat
    # CASE 2: Individual axis
    elif isinstance(ax, plt.Axes):
        axes = [ax]
    # CASE 3: List of axes
    else:
        axes = ax
    assert axes, "Unexpected errpr! 1+ Axes should have been returned here"
    assert isinstance(axes[0], plt.Axes), f"Invalid type of axis: `{type(axes[0])}`"

    # Handle list of axes
    # NOTE: In reality these are all on the same plot, so only change x-axis and ticks
    if len(axes) > 1:
        for curr_ax in axes[:-1]:
            post_plot_logic(
                ax=curr_ax,
                x_lim=x_lim, y_lim=y_lim,
                tick_params=tick_params
            )

    # Save plot
    ax = post_plot_logic(
        ax=axes[-1],
        title=title, title_size=title_size,
        xlabel=xlabel, ylabel=ylabel,
        x_lim=x_lim, y_lim=y_lim,
        tick_params=tick_params,
        legend=legend,
        horizontal_legend=horizontal_legend,
        save_dir=save_dir, save_fname=save_fname,
    )
    return ax


def barplot_with_ci(data, x, y, yerr_low, yerr_high, ax=None,
                    **plot_kwargs):
    """
    Create bar plot with custom confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    x : str
        Name of primary column to group by
    y : str
        Name of column with bar values
    yerr_low : str
        Name of column with lower bound on confidence interval
    yerr_high : str
        Name of column with upper bound on confidence interval
    ax : matplotlib.pyplot.Axis, optional
        If provided, draw plot into this Axis instead of creating a new Axis, by
        default None.
    **plot_kwargs : keyword arguments to pass into `matplotlib.pyplot.bar`

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    # Add default capsize if not specified
    if "capsize" not in plot_kwargs:
        plot_kwargs["capsize"] = 5

    # Create figure
    if ax is None:
        _, ax = plt.subplots()

    # Calculate error values
    yerr = [data[y] - data[yerr_low], data[yerr_high] - data[y]]

    # Create bar plot
    ax.bar(
        x=data[x].values,
        height=data[y].values,
        yerr=yerr,
        **plot_kwargs
    )

    # Remove whitespace on the left and right
    ax.set_xlim(left=-0.5, right=len(data[x].unique()) - 0.5)

    return ax


def grouped_barplot_with_ci(
        data, x, y, hue, yerr_low, yerr_high,
        hue_order=None, color=None, xlabel=None, ylabel=None, ax=None, legend=False,
        **plot_kwargs):
    """
    Create grouped bar plot with custom confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    x : str
        Name of primary column to group by
    y : str
        Name of column with bar values
    hue : str
        Name of secondary column to group by
    yerr_low : str
        Name of column with explicit lower bound on confidence interval for `y`
    yerr_high : str
        Name of column with explicity upper bound on confidence interval for `y`
        interval
    hue_order : list, optional
        Explicit order to use for hue groups, by default None
    color : str, optional
        Color to use for bars, by default None
    xlabel : str, optional
        Label for x-axis, by default None
    ylabel : str, optional
        Label for y-axis, by default None
    ax : matplotlib.pyplot.Axis, optional
        If provided, draw plot into this Axis instead of creating a new Axis, by
        default None.
    legend : bool, optional
        If True, add legend to figure, by default False.
    **plot_kwargs : keyword arguments to pass into `matplotlib.pyplot.bar`

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    # Get unique values for x and hue
    x_unique = data[x].unique()
    xticks = np.arange(len(x_unique))
    hue_unique = data[hue].unique()

    # If specified, fix hue order
    if hue_order:
        # Check that hue order is valid
        if len(hue_order) != len(hue_unique):
            raise RuntimeError(
                f"`hue_order` ({len(hue_order)}) does not match the number of hue groups! ({len(hue_unique)})"
            )
        hue_unique = hue_order

    # Bar-specific constants
    offsets = np.arange(len(hue_unique)) - np.arange(len(hue_unique)).mean()
    offsets /= len(hue_unique) + 1.
    width = np.diff(offsets).mean()

    # Create figure
    if ax is None:
        _, ax = plt.subplots()

    # Create bar plot per hue group
    for i, hue_group in enumerate(hue_unique):
        # Get color for hue group
        if color is not None:
            # CASE 1: One color for all bars
            if isinstance(color, str):
                plot_kwargs["color"] = color
            # CASE 2: One color per bar
            elif isinstance(color, list):
                plot_kwargs["color"] = color[i]

        # Filter for data from hue group and compute differences
        df_group = data[data[hue] == hue_group]
        # Calculate error values
        yerr = [df_group[y] - df_group[yerr_low], df_group[yerr_high] - df_group[y]]

        # Create bar plot
        ax.bar(
            x=xticks+offsets[i],
            height=df_group[y].values,
            width=width,
            label="{} {}".format(hue, hue_group),
            yerr=yerr,
            **plot_kwargs)

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set x-axis ticks
    ax.set_xticks(xticks, x_unique)

    if legend:
        ax.legend()

    return ax


def heatmap(data, cmap=None, xticktop=True, transition_kwargs=None, **kwargs):
    """
    Generate a heatmap. Optionally, create transition matrix using `transition_kwargs`.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame containing the columns to be used for the heatmap.
    cmap : str or matplotlib.colors.Colormap, optional
        Matplotlib colormap
    xticktop : bool, optional
        If True, set x-axis ticks on top
    transition_kwargs : dict, optional
        Arguments to construct transition matrix from 2 columns
        y : str
            The column name used for the heatmap's y-axis.
        x : str
            The column name used for the heatmap's x-axis.
        stat : str, optional
        The statistic to compute, either 'proportion' or 'count', by default "proportion".
        order : list, optional
            The specific order of categories for both axes, by default None.
    **kwargs : Any
        Keyword arguments to pass to `sns.heatmap`
    """
    # CASE 1: If y and x are provided, create a transition matrix
    if transition_kwargs:
        assert "x" in transition_kwargs and "y" in transition_kwargs
        x = transition_kwargs["x"]
        y = transition_kwargs["y"]
        order = transition_kwargs.get("order", None)
        stat = transition_kwargs.get("stat", "proportion")
        assert stat in ["proportion", "count"], f"Invalid stat! {stat}"
        normalize = stat == "proportion"
        df_valid_transition = data[[y, x]].value_counts(normalize=normalize).reset_index()
        transition_matrix = df_valid_transition.pivot(index=y, columns=x, values='proportion')
        transition_matrix.columns = transition_matrix.columns.astype(str)
        transition_matrix.index = transition_matrix.index.astype(str)

        # Reorder rows and columns
        if order:
            order = np.array(order).astype(str)
            transition_matrix = transition_matrix.loc[order].loc[:, order[::-1]]
        # Round to percentage
        transition_matrix = (100 * transition_matrix).round()
    else:
        transition_matrix = data

    # Set colormap
    cmap = cmap if cmap is not None else sns.light_palette("#C6D89E", as_cmap=True)

    # Create heatmap
    ax = sns.heatmap(transition_matrix, annot=True, cmap=cmap, robust=True, **kwargs)

    # Set x-axis ticks, if specified
    if xticktop:
        ax.xaxis.tick_top()

    return ax


def spread_plot(
        data, y, x,
        order=None,
        xlabel=None,
        x_lim=None,
        sharex=False,
        title=None,
        legend=False,
        save_path=None,
        dpi=300,
    ):
    """
    Creates a bar plot visualization showing min/max spread around 0,
    with each bar on its own number line.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame containing the columns to be used for the plot.
    y : str
        The column name used for the y-axis labels (the y_vals) to group by
    x : str
        The column name for values to extract the min and max
    order : list, optional
        The specific order of y_vals for the y-axis, by default None.
        If None, the order in the DataFrame is used.
    xlabel : str, optional
        Label for x-axis
    x_lim : float, optional
        Absolute limit for x-axis ticks
    sharex : bool, optional
        If True, share x-axis labels
    title : str, optional
        The title of the plot, by default "Min/Max Change Spread".
    legend : bool, optional
        If True, create legend
    save_path : str, optional
        Path to file to save the plot, by default None.
    dpi : int, optional
        DPI to save the plot at, by default 300

    Returns
    -------
    plt.Figure
        The Figure object for the plot.
    """
    # Validate input columns
    if y not in data.columns or x not in data.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    # Define muted colors using seaborn's color_palette
    muted_neg = "#6E82B5"
    muted_pos = "#C78E72"

    # Get y values
    y_vals = order or data[y].unique().tolist()
    num_y_vals = len(y_vals)

    # Create a figure and a grid of subplots (each row is one item)
    fig, axes = plt.subplots(num_y_vals, 1, sharex=sharex)

    # Ensure axes is always an array, even for a single item
    if num_y_vals == 1:
        axes = [axes]

    # Get maximum value if sharing x-axis
    x_lim_global = x_lim
    if sharex and not x_lim_global:
        x_lim_global = 1.1 * max(abs(data[x].min()), abs(data[x].max()))

    # Plot for each unique value of y
    for i, y_val in enumerate(y_vals):
        ax = axes[i]
        curr_data = data[data[y] == y_val]
        min_value = curr_data[x].min()
        max_value = curr_data[x].max()
        # Set x-axis limits
        x_lim = x_lim_global if sharex else (1.1 * max(abs(min_value), abs(max_value)))
        ax.set_xlim(-x_lim, x_lim)
        # Plot the negative spread
        ax.barh(0, np.abs(min_value), left=min_value, color=muted_neg)
        # Plot the positive spread
        ax.barh(0, max_value, left=0, color=muted_pos)
        # Add the y value label as the ylabel
        ax.set_ylabel(y_val, rotation=0, ha='right', va='center')
        # Hide x-axis tick labels for other subplots, if sharing
        if sharex and i < num_y_vals - 1:
            ax.tick_params(labelbottom=False)
            # Uncomment to remove the bottom spine too
            # ax.spines['bottom'].set_visible(False)
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        # ax.yaxis.set_visible(False)
        # Remove the "box" (spines) around the plot
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # For the last subplot, add an x-axis label
    if xlabel:
        ax.set_xlabel(xlabel)

    # Add a main title to the figure
    if title:
        fig.suptitle(title)

    # Create legend, if specified
    if legend:
        red_patch = mpatches.Patch(color=muted_neg, label='Decrease')
        green_patch = mpatches.Patch(color=muted_pos, label='Increase')
        fig.legend(handles=[red_patch, green_patch], loc='upper right', bbox_to_anchor=(1, 1))

    # Save if specified
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if not sharex:
            plt.tight_layout(h_pad=0.1)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.clf()
        plt.close()
    return fig


def post_plot_logic(
        ax,
        title=None, title_size=None,
        xlabel=None, ylabel=None,
        x_lim=None, y_lim=None,
        tick_params=None,
        legend=False,
        horizontal_legend=False,
        save_dir=None, save_fname=None,
        dpi=300,
    ):
    """
    Perform post plot operations like adding title, labels, and saving

    Parameters
    ----------
    ax : plt.Axis
        Axis to modify
    title : str, optional
        Title for the plot, by default None.
    title_size : int, optional
        Font size for title, by default 17
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    x_lim : tuple, optional
        Tuple specifying the x-axis limits, by default None.
    y_lim : tuple, optional
        Tuple specifying the y-axis limits, by default None.
    tick_params : dict, optional
        Dictionary specifying the tick parameters, by default None
    legend : bool, optional
        Whether to include a legend, by default False.
    save_dir : str, optional
        Directory to save the plot, by default None.
    save_fname : str, optional
        Filename to save the plot, by default None.
    dpi : int, optional
        DPI to save the plot at, by default 600
    """
    # Early return, if no more things to plot
    if ax is None:
        return

    # Add x-axis and y-axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Update tick parameters
    if tick_params:
        ax.tick_params(**tick_params)

    # Limit x/y-axis
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # Add title
    if title is not None:
        title_kwargs = {}
        if title_size is not None:
            title_kwargs["size"] = title_size
        ax.set_title(title, **title_kwargs)

    # If legend specified, add it outside the figure
    if legend:
        if horizontal_legend:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(ax.get_legend_handles_labels()[0]),
                fontsize='x-small',
            )
        else:
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                ncol=1,
            )

    # Return plot, if not saving
    if not save_dir or not save_fname:
        return ax

    # Save if specified
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight", dpi=dpi)

    # Clear figure, after saving
    plt.clf()
    plt.close()


################################################################################
#                               Helper Functions                               #
################################################################################
def split_violin_by_threshold(ax, threshold=0, flip_idx=None, colors=("#6E82B5", "#C78E72"), line=False):
    """
    Color violin plots to the left and right of a threshold.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing violin plot
    threshold : float
        Threshold value to split errors
    flip_idx : list of int
        Indexes for violins where colors should flip
    line : bool
        If True, add line at the threshold
    """
    # Color by error
    colors = list(colors)
    for idx, violin in enumerate(ax.collections):
        vertices = violin.get_paths()[0].vertices
        curr_colors = colors
        if flip_idx and idx in flip_idx:
            curr_colors = colors[::-1]

        # Fill left side
        left_side = vertices[vertices[:, 0] <= threshold]
        ax.fill(left_side[:, 0], left_side[:, 1], color=curr_colors[0], alpha=0.8)

        # Fill right side
        right_side = vertices[vertices[:, 0] > threshold]
        ax.fill(right_side[:, 0], right_side[:, 1], color=curr_colors[1], alpha=0.8)

    # Add threshold line
    if line:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2)


def remove_dict_keys(dictionary, keys):
    """
    Remove multiple keys from a dictionary.

    Parameters
    ----------
    dictionary : dict
        The dictionary from which to remove keys.
    keys : list
        List of keys to remove.
    """
    for key in keys:
        dictionary.pop(key, None)


def add_default_dict_vals(dictionary, **kwargs):
    """
    Add default values to a dictionary for missing keys.

    Parameters
    ----------
    dictionary : dict
        The dictionary to which default values will be added.
    **kwargs : dict
        Keyword arguments representing key-value pairs to add to the dictionary
        if the key is not already present.
    """
    for key, val in kwargs.items():
        if key not in dictionary:
            dictionary[key] = val


def extract_colors(palette, n_colors):
    """
    Extract the first n_colors colors from a seaborn color palette.

    Parameters
    ----------
    palette : str
        Name of seaborn color palette to extract colors from.
    n_colors : int
        Number of colors to extract from the palette.

    Returns
    -------
    list
        List of n_colors hex color codes.
    """
    palette = sns.color_palette("colorblind", n_colors)
    return list(map(convert_rgb_to_hex, palette))


def convert_rgb_to_hex(rgb):
    """
    Convert RGB to hex color code.

    Parameters
    ----------
    rgb : tuple of floats
        RGB values in range [0, 1]

    Returns
    -------
    str
        Hex color code
    """
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def bolden(text):
    """
    Make a string bold in a matplotlib plot.

    Parameters
    ----------
    text : str
        String to be made bold

    Returns
    -------
    str
        String that will be rendered as bold in a matplotlib plot
    """
    # Ensure latex rendering is enabled
    rc("text", usetex=True)
    return r"\textbf{" + text + r"}"

