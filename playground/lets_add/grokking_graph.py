from grokking import *
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


# === Helper Functions

# --- Plotting functions
# This is mostly a bunch of over-engineered mess to hack Plotly into producing
# the pretty pictures I want, I recommend not reading too closely unless you
# want Plotly hacking practice
def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def imshow(tensor, xaxis=None, yaxis=None, animation_name='Snapshot', **kwargs):
    tensor = torch.squeeze(tensor)
    px.imshow(to_numpy(tensor, flat=False),
              labels={'x':xaxis, 'y':yaxis, 'animation_name':animation_name},
              **kwargs).show()


# Set default colour scheme
imshow = partial(imshow, color_continuous_scale='Blues')
# Creates good defaults for showing divergent colour scales (ie with both
# positive and negative values, where 0 is white)
imshow_div = partial(imshow, color_continuous_scale='RdBu', color_continuous_midpoint=0.0)
# Presets a bunch of defaults to imshow to make it suitable for showing heatmaps
# of activations with x axis being input 1 and y axis being input 2.
inputs_heatmap = partial(imshow, xaxis='Input 1', yaxis='Input 2', color_continuous_scale='RdBu', color_continuous_midpoint=0.0)


def line(x, y=None, hover=None, xaxis='', yaxis='', **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x=to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()


def scatter(x, y, **kwargs):
    px.scatter(x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), **kwargs).show()


def histogram(x, nbins=100, **kwargs):
    px.histogram(to_numpy(x), nbins=nbins, **kwargs).show()


def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()


def line_marker(x, **kwargs):
    lines([x], mode='lines+markers', **kwargs)


def animate_lines(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, xaxis='x', yaxis='y', **kwargs):
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    px.line(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover,**kwargs).show()


def imshow_fourier(tensor, title='', animation_name='snapshot', facet_labels=[], **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    if tensor.shape[0]==p*p:
        tensor = unflatten_first(tensor)
    tensor = torch.squeeze(tensor)
    fig=px.imshow(to_numpy(tensor),
            x=fourier_basis_names,
            y=fourier_basis_names,
            labels={'x':'x Component',
                    'y':'y Component',
                    'animation_frame':animation_name},
            title=title,
            color_continuous_midpoint=0.,
            color_continuous_scale='RdBu',
            **kwargs)
    fig.update(data=[{'hovertemplate':"%{x}x * %{y}yValue:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show()


def animate_multi_lines(lines_list, y_index=None, snapshot_index = None, snapshot='snapshot', hover=None, swap_y_animate=False, **kwargs):
    # Can plot an animation of lines with multiple lines on the plot.
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j])+[snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index+[snapshot, 'x'])
    px.line(df, x='x', y=y_index, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs).show()


def animate_scatter(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, yaxis='y', xaxis='x', color=None, color_name = 'color', **kwargs):
    # Can plot an animated scatter plot
    # lines_list has shape snapshot x 2 x line
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if type(color)==torch.Tensor:
        color = to_numpy(color)
    if len(color.shape)==1:
        color = einops.repeat(color, 'x -> snapshot x', snapshot=lines_list.shape[0])
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append([lines_list[i, 0, j].item(), lines_list[i, 1, j].item(), snapshot_index[i], color[i, j]])
    print([lines_list[:, 0].min(), lines_list[:, 0].max()])
    print([lines_list[:, 1].min(), lines_list[:, 1].max()])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    px.scatter(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()], range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()], hover_name=hover, color=color_name, **kwargs).show()
