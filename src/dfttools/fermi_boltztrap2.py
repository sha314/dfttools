# Computes energy bands, group velocities and inverse of effective mass 
# using boltztrap interpolation along given symmetry path
# and plots it


from dfttools import compute, units
from dfttools.units import *
import numpy as np
import ase.dft.kpoints as asekp
import ast
import itertools



from BoltzTraP2.misc import TimerContext
from BoltzTraP2 import fite
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio


def get_hexagonal_cell_edges_v2(bg, scale=[1,1,1]):
    """
    Compute first Brillouin zone boundary edges and vertices.

    Parameters
    ----------
    bg : ndarray, shape (3, 3)
        Reciprocal lattice vectors as rows: bg = [
                                                    [b1], 
                                                    [b2], 
                                                    [b3]]
    scale : when you need to scale the hexagonal cell along any direction, use scale factors as [kx,ky,kz] format

    Returns
    -------
    edge_segments : list of dict
        Each dict has keys 'x', 'y', 'z' with two-element lists. Order : vertical, bottom, top
    all_vertices : ndarray, shape (12, 3)
        All BZ vertices (hexagonal prism). Order : bottom, top
    """
    b1 = bg[0]*scale[0]
    b2 = bg[1]*scale[1]
    b3 = bg[2]*scale[2]

    # --- Detect hexagonal symmetry ---
    cos_12 = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
    cos_13 = np.dot(b1, b3) / (np.linalg.norm(b1) * np.linalg.norm(b3))
    cos_23 = np.dot(b2, b3) / (np.linalg.norm(b2) * np.linalg.norm(b3))

    is_hexagonal = (
        np.abs(np.abs(cos_12) - 0.5) < 0.15 and
        np.abs(cos_13) < 0.15 and
        np.abs(cos_23) < 0.15
    )
    if not is_hexagonal:
        raise ValueError(f"Not hexagonal: cos12={cos_12:.3f}, cos13={cos_13:.3f}, cos23={cos_23:.3f}")
        pass

    # --- Generate neighbors ---
    neighbors = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                if i == 0 and j == 0 and k == 0:
                    continue
                neighbors.append(i * b1 + j * b2 + k * b3)
    neighbors = np.array(neighbors)

    # --- Find 6 shortest in-plane G vectors ---
    b3_hat = b3 / np.linalg.norm(b3)
    G_in_plane = [(np.linalg.norm(G), G) for G in neighbors
                  if np.abs(np.dot(G, b3_hat)) < 0.1 * np.linalg.norm(b3)]
    G_in_plane.sort(key=lambda x: x[0])

    hex_G = [g[1] for g in G_in_plane[:6]]
    print("hex_G ", hex_G)
    # Sort by angle around b3 axis using a proper plane basis
    center_plane = np.mean(hex_G, axis=0)
    print("center_plane ", center_plane)
    e3 = b3_hat
    ref = np.array([1.0, 0.0, 0.0]) if np.abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, e3) * e3
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    print(e1, e2, e3)

    angles = [np.arctan2(np.dot(g - center_plane, e2), np.dot(g - center_plane, e1))
              for g in hex_G]
    print(angles)
    hex_G = [g for _, g in sorted(zip(angles, hex_G))]

    # --- Build hexagonal prism ---
    hex_vertices = np.array([(hex_G[i] + hex_G[(i + 1) % 6]) / 3 for i in range(6)])
    b3_half = 0.5 * b3
    bottom = hex_vertices - b3_half
    top    = hex_vertices + b3_half
    all_vertices = np.vstack([bottom, top])

    edge_segments = []
    # Vertical edges
    for i in range(6):
        edge_segments.append({'x': [bottom[i,0], top[i,0]],
                              'y': [bottom[i,1], top[i,1]],
                              'z': [bottom[i,2], top[i,2]]})
    # Bottom & top hexagons
    for i in range(6):
        j = (i + 1) % 6
        edge_segments.append({'x': [bottom[i,0], bottom[j,0]],
                              'y': [bottom[i,1], bottom[j,1]],
                              'z': [bottom[i,2], bottom[j,2]]})
        edge_segments.append({'x': [top[i,0], top[j,0]],
                              'y': [top[i,1], top[j,1]],
                              'z': [top[i,2], top[j,2]]})

    return edge_segments, all_vertices



    
def get_high_symmetry_points(bg, scale=[1,1,1]):
    """
    bg : Reciprocal lattice vectors as rows
    scale : you can scale up or down along any vectors.
    Return Γ, K, M, A, H, L for a hexagonal BZ.
    """

    b1, b2, b3 = bg[0], bg[1], bg[2]
    edge_segments, hex_vertices = get_hexagonal_cell_edges_v2(bg)
    b3_half = 0.5 * b3

    return {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'K': hex_vertices[0],
        'M': 0.5 * (hex_vertices[0] + hex_vertices[1]),
        'A': b3_half,
        'H': hex_vertices[0] + b3_half,
        'L': 0.5 * (hex_vertices[0] + hex_vertices[1]) + b3_half,
    }


def frac_to_cartesian(k_frac, bg):
    """
    Convert fractional k-points to Cartesian coordinates.
    
    Parameters
    ----------
    k_frac : ndarray, shape (3,) or (3, N)
        Fractional coordinates (kx, ky, kz) in [0, 1).
    bg : ndarray, shape (3, 3)
        Reciprocal lattice vectors as rows: bg = [b1, b2, b3].

    Operation performed is simply 'k_frac[0]*bg[0] + k_frac[1]*bg[1] + k_frac[2]*bg[2]' but with @ 
    
    Returns
    -------
    k_cart : ndarray, same shape as k_frac
    """
    return bg.T @ k_frac




def rotate_z(KX_data, KY_data, angle):
    """
    angle : angle in degree
    """
    theta = np.radians(angle)
    c = np.cos(theta)
    s = np.sin(theta)
    
    R = np.array([
        [ c, -s],
        [ s,  c],
    ])
    KX = KX_data.ravel()
    KY = KY_data.ravel()
    Kvec = np.vstack([KX, KY])
    # print(Kvec.shape)
    Kvec_rot = R @ Kvec

    KX_rot = Kvec_rot[0, :].reshape(KX_data.shape)
    KY_rot = Kvec_rot[1, :].reshape(KY_data.shape)

    return KX_rot, KY_rot



def rotate_plain_z(KX_data, KY_data, angle):
    """
    KX_data : 1D array
    KY_data : 1D array
    angle : angle in degree
    """
    theta = np.radians(angle)
    c = np.cos(theta)
    s = np.sin(theta)
    
    R = np.array([
        [ c, -s],
        [ s,  c],
    ])

    Kvec = np.vstack([KX_data, KY_data])
    print(Kvec.shape)
    Kvec_rot = R @ Kvec
    print(Kvec_rot.shape)
    

    return Kvec_rot[0], Kvec_rot[1]



def get_uniform_k_grid_interpolate(
        data, equivalences, coeffs,
    nk1, nk2, nk3, K_grid,
    band_ids=None,
):
    """
    Extract band energies along a high-symmetry k-path from BoltzTraP2 data.

    Uses BoltzTraP2's Fourier interpolation to compute energies at arbitrary
    k-points along the requested path. And returned energy, velocity and curvature grids are reshaped with specified nk points

    Parameters
    ----------
    data, equivalences, coeffs : obtained from ``load_interpolation``.

    nk1, nk2, nk3 : 
    K_grid        : k_grid in fractional coordinate

    band_ids : list[int] or None, optional
        Subset of band indices to extract. If None, all bands are used.

    Returns
    -------
    kpoints_grid : list[np.ndarray]
        List of dense 3D k-point arrays, one per segment.
        Shape of each array: ``(n_k, 3)``.

    energy_grid : list[np.ndarray]
        List of energy arrays, one per segment.
        Shape of each array: ``(n_bands, nk1, nk2, nk3)``.

    velocity_grid : 
        Group velocity in [Ha . Bohr] unit.
        Shape of each array: ``(n_bands, 3, nk1, nk2, nk3)``.

    curvature_grid : 
        Band Curvature in [Ha . Bohr^2] unit. Also known as inverse of effective mass.
        Shape of each array: ``(n_bands, 3, 3, nk1, nk2, nk3)``.
    """

    coeffs_tmp = coeffs
    if band_ids is not None:
        coeffs_tmp = coeffs[band_ids,:]
        pass

    with TimerContext() as timer:
        egrid, vgrid, cgrid = fite.getBands(
            K_grid, equivalences, data.get_lattvec(), coeffs_tmp, curvature=True
        )
        deltat = timer.get_deltat()
        print("rebuilding the bands took {:.3g} s".format(deltat))
        pass
    energy_grid = egrid.reshape((-1,nk1,nk2,nk3))
    velocity_grid = vgrid.reshape((-1,3,nk1,nk2,nk3))
    invmass_grid = cgrid.reshape((-1,3,3,nk1,nk2,nk3))
    return K_grid, energy_grid, velocity_grid, invmass_grid


def get_uniform_nk_interpolate(
        data, equivalences, coeffs,
    nk1, nk2, nk3,
    band_ids=None,
):
    """
    same as ```get_uniform_k_grid_interpolate```
    but it does not take k_grid as input, rather it calculates in range (0,1) and passes it to ```get_uniform_k_grid_interpolate```
    """


    k1 = np.linspace(0, 1, nk1)
    k2 = np.linspace(0, 1, nk2)
    k3 = np.linspace(0, 1, nk3)
    K1, K2, K3 = np.meshgrid(k1, k2, k3)
    K_grid = np.column_stack((K1.ravel(), K2.ravel(), K3.ravel()))

    return get_uniform_k_grid_interpolate(data, equivalences, coeffs, nk1, nk2, nk3, K_grid, band_ids)
    



def plot_E_kx_ky(KX_bz, KY_bz, Energy, surface_color, color_range=(-1,1), 
                 at_kz=0, fig=None, edges=None, plane_at_z=None,
                 high_sym_points_dots=True, high_sym_points_label=True,
                 color_bar_dict={
    'text' : (
                        '(1/<i>m</i><sub>xx</sub><sup>*</sup> + '
                        '1/<i>m</i><sub>yy</sub><sup>*</sup>)'
                        '<i>m</i><sub>e</sub>/2 '
                    ),
    'tickvals' : [-1, -0.5, 0, 0.5, 1],
                },
                colorscale=[
                            [0.0, "rgb(0, 0, 180)"],      # deep blue
                            [0.35, "rgb(120, 170, 255)"], # light blue
                            [0.5, "rgb(255, 255, 255)"],  # white at center
                            [0.65, "rgb(255, 120, 120)"], # light red
                            [1.0, "rgb(180, 0, 0)"]       # deep red
                    ]
                    ):
    """
    KX_bz, KY_bz  : (nk1,nk2) shapred array
    Energy        : of a single band shaped (nk1,nk2)
    surface_color : color scheme of a single band shapred (nk1,nk2)
    colorscale    : default [
                            [0.0, "rgb(0, 0, 180)"],      # deep blue
                            [0.35, "rgb(120, 170, 255)"], # light blue
                            [0.5, "rgb(255, 255, 255)"],  # white at center
                            [0.65, "rgb(255, 120, 120)"], # light red
                            [1.0, "rgb(180, 0, 0)"]       # deep red
                    ]
                    you can use jet, viridis and any color in plotly.express.colors.named_colorscales()
    
    """
    if fig is None:
        fig = go.Figure()
        pass


    # KX_bz, KY_bz = rotate_z(KX_bz, KY_bz, 30)
    for i in range(6):
        fig.add_trace(go.Surface(
            x=KX_bz, y=KY_bz, z=Energy,
            name='Band',
            surfacecolor=surface_color,   # color = some other function
            # colorscale='RdBu_r',
            colorscale=[
                [0.0, "rgb(0, 0, 180)"],      # deep blue
                [0.35, "rgb(120, 170, 255)"], # light blue
                [0.5, "rgb(255, 255, 255)"],  # white at center
                [0.65, "rgb(255, 120, 120)"], # light red
                [1.0, "rgb(180, 0, 0)"]       # deep red
        ],
            showscale=(i==0),
            # colorbar=dict(title=r'$(\frac{1}{m^*_{xx}} + \frac{1}{m^*_{yy}})/2$', x=0.9),
            # colorbar=dict(title=r'(1/mxx + 1/myy)/2', x=0.9),
            # colorbar=dict(
            #     title=dict(text=r'$(\frac{1}{m^*_{xx}} + \frac{1}{m^*_{yy}})/2$', font=dict(size=14)),
            #     titleside='right'
            # ),
            colorbar=dict(
                title=dict(
                    text=color_bar_dict['text'],
                    side='right',
                    font=dict(
                        family='Times New Roman',
                        size=20,
                        color='black',
                    ),
                ),

                # Position
                x=0.92,
                y=0.50,

                # Size
                len=0.72,
                thickness=20,

                xanchor='left',
                yanchor='middle',

                # Ticks
                tickfont=dict(
                    family='Times New Roman',
                    size=20,
                    color='black',
                ),

                tickmode='array',
                # tickvals=[-1, -0.5, 0, 0.5, 1],
                # ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0'],

                tickvals=[a for a in np.linspace(color_range[0], color_range[1], 5)],
                ticktext=[f"{a:.2f}" for a in np.linspace(color_range[0], color_range[1], 5)],

                outlinewidth=1,
                outlinecolor='black',
            ),
            cmin=color_range[0],    # hides surface where z < -2
            cmax=color_range[1],     # hides surface where z > 2
            opacity=1.0,
            # Remove mesh lines
            hidesurface=False,
            contours=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False),
            ),
        ))
        # fig.add_trace(go.Surface(
        #     x=KX_bz, y=KY_bz, z=np.zeros_like(Energy),
        #     name='E_F',
        #     colorscale=[[0, 'rgba(128, 128, 128, 0.6)'], [1, 'rgba(128, 128, 128, 0.6)']],  # black, 30% alpha
        #     showscale=False,
        #     hoverinfo='skip',
        # ))

        KX_bz, KY_bz = rotate_z(KX_bz, KY_bz, 60)
        pass


    # fig = go.Figure(data=[
    #     go.Surface(
    #         x=KX_bz,           # 2D array or 1D kx values
    #         y=KY_bz,           # 2D array or 1D ky values  
    #         z=Energy,      # must be 2D with shape (len(y), len(x))
    #         colorscale='RdBu_r',
    #         colorbar=dict(title='E (eV)'),
    #     )
    # ])


    if edges is not None:
        # edges = edges[7:18:2]

        x_edges, y_edges, z_edges = [], [], []
        for seg in edges:
            x_edges.extend(seg['x'] + [None])
            y_edges.extend(seg['y'] + [None])
            z_edges.extend(seg['z'] + [None])


        fig.add_trace(
            go.Scatter3d(
                x=x_edges,
                y=y_edges,
                z=z_edges,

                mode='lines',

                line=dict(
                    color='black',
                    width=5,
                ),

                hoverinfo='skip',
                showlegend=False,
            )
        )
        pass



    ########## Gamma, M and K point
    if high_sym_points_dots:
        x_val, y_val, z_val = 0, 0, 0.017
        z_val = 0
        fig.add_trace(
            go.Scatter3d(
                x=[x_val],
                y=[y_val],
                z=[z_val],
                mode='markers',
                marker=dict(size=5, color='black'),
                hoverinfo='skip',
                showlegend=False,
            )
        )

        # Add the 3D annotation via layout update
        annotation_dict_G = dict(
                        x=x_val,
                        y=y_val,
                        z=z_val,
                        text='<b>Γ</b>',
                        showarrow=False,
                        font=dict(size=25, color='black'),
                        # Pixel offsets to bring the label closer (negative moves it up)
                        yshift=6,   # adjust this value
                        xshift=-8,
                    )


        x_val, y_val = 0, 0.2
        
        fig.add_trace(
            go.Scatter3d(
                x=[x_val],
                y=[y_val],
                z=[z_val],
                mode='markers',
                marker=dict(size=5, color='black'),        
                hoverinfo='skip',
                showlegend=False,
            )
        )
        annotation_dict_M = dict(
                        x=x_val,
                        y=y_val,
                        z=z_val,
                        text='<b>M</b>',
                        showarrow=False,
                        font=dict(size=25, color='black'),
                        # Pixel offsets to bring the label closer (negative moves it up)
                        yshift=8,   # adjust this value
                        xshift=20,
                    )


        x_val=-0.1156
        fig.add_trace(
            go.Scatter3d(
                x=[x_val],
                y=[y_val],
                z=[z_val],
                mode='markers',
                marker=dict(size=5, color='black'),
                hoverinfo='skip',
                showlegend=False,
            )
        )
        annotation_dict_K = dict(
                        x=x_val,
                        y=y_val,
                        z=z_val,
                        text='<b>K</b>',
                        showarrow=False,
                        font=dict(size=25, color='black'),
                        # Pixel offsets to bring the label closer (negative moves it up)
                        yshift=8,   # adjust this value
                        xshift=15,
                    )

        if high_sym_points_label:
            fig.update_layout(
                scene=dict(
                    annotations=[
                        annotation_dict_G, annotation_dict_M, annotation_dict_K
                    ]
                )
            )


    theta=np.radians(30)
    viewx, viewy = np.array([np.cos(theta), np.sin(np.radians(180)-theta)]) * 1.9
    fig.update_layout(

        # ==========================================================
        # Overall figure
        # ==========================================================
        width=900,
        height=700,

        paper_bgcolor='white',
        plot_bgcolor='white',

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=20,
        ),

        font=dict(
            family='Times New Roman',
            size=18,
            color='black',
        ),

        # ==========================================================
        # 3D scene
        # ==========================================================
        scene=dict(

            # ------------------------------------------------------
            # X axis
            # ------------------------------------------------------
            xaxis=dict(
                title=dict(
                    text='<i>k</i><sub>x</sub>',
                    font=dict(
                        family='Times New Roman',
                        size=28,
                        color='black',
                    ),
                ),

                range=[-0.3, 0.3],
                autorange=False,

                tickfont=dict(
                    family='Times New Roman',
                    size=16,
                    color='black',
                ),

                nticks=7,

                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,

                showline=True,
                linecolor='black',
                linewidth=2,

                zeroline=False,

                ticks='outside',
                ticklen=5,
                tickwidth=1.5,

                backgroundcolor='white',
            ),

            # ------------------------------------------------------
            # Y axis
            # ------------------------------------------------------
            yaxis=dict(
                title=dict(
                    text='<i>k</i><sub>y</sub>',
                    font=dict(
                        family='Times New Roman',
                        size=28,
                        color='black',
                    ),
                ),

                range=[-0.3, 0.3],
                autorange=False,

                tickfont=dict(
                    family='Times New Roman',
                    size=16,
                    color='black',
                ),

                nticks=7,

                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,

                showline=True,
                linecolor='black',
                linewidth=2,

                zeroline=False,

                ticks='outside',
                ticklen=5,
                tickwidth=1.5,

                backgroundcolor='white',
            ),

            # ------------------------------------------------------
            # Z axis
            # ------------------------------------------------------
            zaxis=dict(
                title=dict(
                    text='',
                    font=dict(
                        family='Times New Roman',
                        size=22,
                        color='black',
                    ),
                ),

                range=[-0.25, 0.15],
                autorange=False,

                tickfont=dict(
                    family='Times New Roman',
                    size=15,
                    color='black',
                ),

                nticks=6,

                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,

                showline=True,
                linecolor='black',
                linewidth=2,

                zeroline=False,

                ticks='outside',
                ticklen=5,
                tickwidth=1.5,

                backgroundcolor='white',
            ),

            # ------------------------------------------------------
            # Physical / visual aspect ratio
            # ------------------------------------------------------
            aspectmode='manual',

            aspectratio=dict(
                x=1,
                y=1,
                z=0.7,
            ),

            # ------------------------------------------------------
            # Camera
            # ------------------------------------------------------
            camera=dict(
                eye=dict(
                    x=viewx,
                    y=viewy,
                    z=1.3,
                ),

                center=dict(
                    x=0,
                    y=0,
                    z=0,
                ),

                up=dict(
                    x=0,
                    y=0,
                    z=1,
                ),
            ),

            bgcolor='white',
        ),
    )

    ############ Plotting a plane at z=0
    if plane_at_z is not None:
        xplane = np.array([
            [-0.3,  0.3],
            [-0.3,  0.3]
        ])

        yplane = np.array([
            [-0.3, -0.3],
            [ 0.3,  0.3]
        ])

        zplane = np.ones_like(xplane)*plane_at_z

        fig.add_trace(
            go.Surface(
                x=xplane,
                y=yplane,
                z=zplane,

                colorscale=[
                    [0, 'rgba(160,160,160,0.6)'],
                    [1, 'rgba(160,160,160,0.6)']
                ],

                showscale=False,
                hoverinfo='skip',
                name='E = 0',

                contours={
                'x': {'show': True, 'color': 'black', 'width': 1, 'start': -5, 'end': 5, 'size': 1},
                'y': {'show': True, 'color': 'black', 'width': 1, 'start': -5, 'end': 5, 'size': 1},
            },
            )
        )


    ######### Update z annotation
    fig.add_annotation(
        text='<i>E</i> (eV)',
        x=0.03,
        y=0.50,
        xref='paper',
        yref='paper',
        textangle=-90,
        showarrow=False,
        font=dict(
            family='Times New Roman',
            size=26,
            color='black',
        ),
    )

    fig.add_annotation(
        text=f'<i>kz={at_kz:.4f}</i>',
        x=0.9,
        y=0.70,
        xref='paper',
        yref='paper',
        textangle=-90,
        showarrow=False,
        font=dict(
            family='Times New Roman',
            size=26,
            color='black',
        ),
    )

    # 1. Interactive HTML (recommended — keeps zoom/rotate)
    # fig.write_html("in-plane-energy.html")

    # 2. Static image (PNG, PDF, SVG, JPEG)
    # Requires: pip install kaleido
    # filename = f"Nb3S4-E(kx,ky)-plotly-2c-kz{at_kz:.4f}.png"
    # print(filename)
    # fig.write_image(filename, scale=2)   # 2x resolution

    # fig.show()
    return fig

def plot_edges():


    pass


def plot_E_kx_ky_btp(dft_data, bt2filnam, niter, band_ids, nk1, nk2, nk3, ikz=0):
    """
    dft_data, bt2filnam, niter : check load_interpolation()

    band_ids : tuple of bands to be plotted
    nk1, nk2 : number of k-points along k1 and k2 axes for interpolation
    at_kz    : the kz slice 
    
    """
    data, equivalences, coeffs = compute.load_interpolation(dft_data, bt2filnam, niter)
    lattvec = data.get_lattvec()
    reciprocal_lattice_vec = np.linalg.inv(lattvec)*2*np.pi
    bg = reciprocal_lattice_vec

    k1 = np.linspace(0, 1.0, nk1)
    k2 = np.linspace(0, 1.0, nk2)
    k3 = np.linspace(0, 1.0, nk3)
    KX_frac, KY_frac, KZ_frac = np.meshgrid(k1, k2, k3, indexing='ij')

    k_frac_grid = np.column_stack((KX_frac.ravel(), KY_frac.ravel(), KZ_frac.ravel()))

    K_grid, energy_grid, velocity_grid, curvature_grid = get_uniform_k_grid_interpolate(
        data, equivalences, coeffs,
        nk1, nk2, nk3, k_frac_grid,
        band_ids=band_ids,)


    iband=[i for i in range(len(band_ids))]

    mat_xx = curvature_grid[iband,0,0,:,:,ikz]
    mat_yy = curvature_grid[iband,1,1,:,:,ikz]
    imass = (mat_xx + mat_yy)/2.


    k_bz_grid = frac_to_cartesian(k_frac_grid.T, bg)


    KX_bz = k_bz_grid[0, :].reshape(KX_frac.shape)
    KY_bz = k_bz_grid[1, :].reshape(KY_frac.shape)


    Energy = (energy_grid[iband,:,:,ikz]-data.fermi)/units.eV

    fig = plot_E_kx_ky(KX_bz, KY_bz, Energy, imass, at_kz=k_bz_grid[2,ikz])


    return fig


def generate_band_path(kpath, cell, nkpoints):
    """
    
    
    """
    band_path = asekp.bandpath(kpath, cell, nkpoints)
    if isinstance(band_path, asekp.BandPath):
        # For newer versions of ASE.
        kp = band_path.kpts
        # print("band_path.get_linear_kpoint_axis()")
        # print(band_path.get_linear_kpoint_axis())
        dkp, dcl = band_path.get_linear_kpoint_axis()[:2]
        # print("dkp, dcl")
        # print(dkp, dcl)
    else:
        # For older versions of ASE.
        kp, dkp, dcl = band_path
        pass
    return kp, dkp, dcl


def parse_one_k_path(kpath="[0.0,0.0,0.0], [0.5, 0.0, 0.0]"):
    """
    kpath : str
        k path as str in fractional coordiante.
        It can be two points coordinte: '[0.0,0.0,0.0], [0.5, 0.0, 0.0]'
        Or multiple points coordinates: '[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.0,0.0,0.0], [0.0,0.0,0.5]'
    """
    try:
        kpaths = ast.literal_eval(kpath)
    except ValueError:
        print("cannot be parsed as a Python literal")

    kpaths = [
        list(group)
        for k, group in itertools.groupby(kpaths, key=lambda x: x is not None)
        if k
    ]

    # print(kpaths)
    kpaths = [np.array(i, dtype=np.float64) for i in kpaths]  
    return kpaths




def parse_k_path(kpath):
    # The second position alargument is first interpreted as a Python literal,
    # and after parsing it is cast to a NumPy array, which must have the right
    # dimensions. The special value None directs the parser to split the path
    # in several parts.

    kpath_list = []
    if isinstance(kpath, list) or isinstance(kpath, tuple):
        # each element of list is a k path, evaluate them independently
        kpath_list = [parse_one_k_path(kp) for kp in kpath]
    else:
        kpath_list = [parse_one_k_path(kpath)]
        pass
    return kpath_list



def extract_kpath_data(
    kpts_grid,           # (N_kpts, 3)  – your uniform grid k-points (fractional)
    energies_grid,       # (N_bands, N_kpts) – energies at those k-points
    cell,                # ASE cell or 3×3 array
    kpath,          # tuple(kpath_string, nkpoints_list) or None for default
    nkpoints_list,
    fermi=0.0,
    shift_eV=0.0,
    tol=1e-5,
):
    """
    Extract band-structure data along a k-path from a uniform energy grid.

    Returns
    -------
    segments_info : list[dict]
        [{"label": str, "first": int, "last": int}, ...]
        Indices refer to a flattened concatenation of all segments.
    kpaths_list : list[np.ndarray]
        [array(n_points, 3), ...] - 3D k-points along each segment
    energies_list : list[np.ndarray]
        [array(n_bands, n_points), ...] - energies (or any function of 3D k points) for each segment
    """

    n_bands = energies_grid.shape[0]
    e_shift = fermi + shift_eV / Ha_to_eV
    energies = energies_grid - e_shift


    kpaths_raw = ast.literal_eval(kpath)
    print(type(kpaths_raw))
    if not (isinstance(kpaths_raw, list) or isinstance(kpaths_raw, tuple)):
        raise ValueError("kpath must parse to a Python list")

    # Split into segments (by None if present, else consecutive pairs)
    has_none = any(x is None for x in kpaths_raw)
    if has_none:
        kpath_segments = [
            list(g) for k, g in itertools.groupby(kpaths_raw, key=lambda x: x is not None) if k
        ]
    else:
        kpath_segments = [
            [kpaths_raw[i], kpaths_raw[i + 1]]
            for i in range(len(kpaths_raw) - 1)
        ]

    kpath_segments = [np.array(seg, dtype=np.float64) for seg in kpath_segments]
    n_seg = len(kpath_segments)

    # Ensure nkpoints_list matches number of segments
    if len(nkpoints_list) < n_seg:
        nkpoints_list = np.concatenate([
            nkpoints_list,
            np.full(n_seg - len(nkpoints_list), nkpoints_list[-1])
        ])
    else:
        nkpoints_list = nkpoints_list[:n_seg]

    # --- 3. Grid lookup tables --------------------------------------------
    kpts_grid = np.asarray(kpts_grid).reshape(-1, 3)
    kpts_wrapped = kpts_grid % 1.0

    kpaths_list = []
    energies_list = []

    # --- 4. Loop over segments --------------------------------------------
    for iseg, seg in enumerate(kpath_segments):
        n_kp = int(nkpoints_list[iseg])

        # Generate dense k-path with ASE
        kp_dense, _, _ = generate_band_path(seg, cell, n_kp)

        n_points = len(kp_dense)

        # Map dense k-points → nearest grid point
        kp_dense_wrapped = kp_dense % 1.0
        e_seg = np.zeros((n_bands, n_points))

        for i_kp, kp in enumerate(kp_dense_wrapped):
            diff = kpts_wrapped - kp
            diff -= np.rint(diff)                 # minimum image for periodicity
            dists = np.linalg.norm(diff, axis=1)
            idx = np.argmin(dists)

            # if dists[idx] > tol:
            #     print(f"Warning: k-point {kp} matched to grid with distance {dists[idx]:.2e}")

            e_seg[:, i_kp] = energies[:, idx]

        kpaths_list.append(kp_dense.copy())
        energies_list.append(e_seg)
        pass


    return kpaths_list, energies_list


    


def extract_kpath_interpolate(
        data, equivalences, coeffs,
    kpath,          # tuple(kpath_string, nkpoints_list) or None for default
    nkpoints_list,
    band_ids=None,
):
    """
    Extract band energies along a high-symmetry k-path from BoltzTraP2 data.

    Uses BoltzTraP2's Fourier interpolation to compute energies at arbitrary
    k-points along the requested path.

    Parameters
    ----------
    data : boltztrap2.BztInterpolatorData
        Object containing lattice vectors, Fermi level, etc.
    equivalences : list
        Symmetry equivalences from ``load_interpolation``.
    coeffs : np.ndarray
        Interpolation coefficients from ``load_interpolation``.
    kpath_list : str or list of str
        Custom k-path. Two format (1) A single str with all k-paths (2) list of k-path segments as str
    nkpoints_list : list[int]
        Number of data points along each k-path
        
    band_ids : list[int] or None, optional
        Subset of band indices to extract. If None, all bands are used.

    Returns
    -------
    kpoints_grid : list[np.ndarray]
        List of dense 3D k-point arrays, one per segment.
        Shape of each array: ``(n_k, 3)``.

    energy_grid : list[np.ndarray]
        List of energy arrays, one per segment.
        Shape of each array: ``(n_bands, n_k)``.

    velocity_grid : 
        Group velocity in [Ha . Bohr] unit.
        Shape of each array: ``(3, n_bands, n_k)``.

    curvature_grid : 
        Band Curvature in [Ha . Bohr^2] unit. Also known as inverse of effective mass.
        Shape of each array: ``(3, 3, n_bands, n_k)``.
    """

    energy_grid = []
    velocity_grid = []
    curvature_grid = []
    kpoints_grid = []

    coeffs_tmp = coeffs
    if band_ids is not None:
        coeffs_tmp = coeffs[band_ids,:]
        pass

    kpaths = parse_k_path(kpath)
    
    for ikpath, kpath in enumerate(kpaths):
        print("k path #{}".format(ikpath + 1))
        # Generate the explicit point list.
        kp, dkp, dcl = generate_band_path(kpath, data.atoms.cell, nkpoints_list[ikpath])
        # print("band path ", kp)
        kpoints_grid.append(kp)
        # Compute the band energies
        with TimerContext() as timer:
            egrid, vgrid, cgrid = fite.getBands(
                kp, equivalences, data.get_lattvec(), coeffs_tmp, curvature=True
            )
            deltat = timer.get_deltat()
            print("rebuilding the bands took {:.3g} s".format(deltat))
        energy_grid.append(egrid)
        velocity_grid.append(vgrid)
        curvature_grid.append(cgrid)
        pass

    return kpoints_grid, energy_grid, velocity_grid, curvature_grid




def plot_and_save_bands_velocity_imass(filename, energy, velocity, curvature, ib, nkpoints_list, labels, efermi, band_color='blue'):
    fig, axes = plt.subplots(3, len(energy), figsize=(15, 6), sharey="row", gridspec_kw={
        "width_ratios": nkpoints_list,
        "wspace":0,
        "hspace":0
        }, dpi=300)

    for k in range(len(energy)):
        x = np.linspace(0, 1, energy[k].shape[1])
        ax = axes[0, k]
        ax.plot(x, energy[k][ib].T-efermi, label=f"E(K),{ib}", color=band_color)
        ax.axhline(0, 0, 10, color='k', linestyle=":")
        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            ax.set_ylabel(r"$E-E_F (Ha)$")
        ax.set_xlabel(f"{labels[k]}")
        ax.set_ylim(-0.015, 0.015)


        ax = axes[1, k]
        ax.plot(x, velocity[k][0, ib].T, label="vx")
        ax.plot(x, velocity[k][1, ib].T, label="vy")
        ax.plot(x, velocity[k][2, ib].T, label="vz")
        ax.set_ylim(-0.15, 0.15)

        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            # ax.set_ylabel(r"$v [Ha\cdot Bohr/\hbar$]")
            ax.set_ylabel(r"$v/v_0$")
        ax.set_xlabel(f"{labels[k]}")


        ax = axes[2, k]

        ax.plot(x, curvature[k][0,0, ib].T, label="xx")
        ax.plot(x, curvature[k][1,1, ib].T, label="yy")
        ax.plot(x, curvature[k][2,2, ib].T, label="zz")
        ax.plot(x, curvature[k][0,1, ib].T, label="xy")
        ax.plot(x, curvature[k][0,2, ib].T, label="xz")
        ax.plot(x, curvature[k][1,2, ib].T, label="yz")
        ax.set_ylim(-0.15, 0.15)

        ax.axhline(0, 0, 10, color='k', linestyle=":")
        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            # ax.set_ylabel(r"$Ha\cdot Bohr^2$")
            # ax.set_ylabel(r"$m_e/m^* [Ha\cdot Bohr^2/\hbar^2$]$")
            ax.set_ylabel(r"$m_e/m^*$")
        ax.set_xlabel(f"{labels[k]}")
        axes[0, k].set_xticks([])
        axes[1, k].set_xticks([])
        axes[2, k].set_xticks([])
        pass
    for ax in axes.flat:
        # Y-axis spine
        ax.spines["left"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0.3)

        # X-axis spine
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["top"].set_linewidth(1.2)

        ax.margins(x=0)           # remove all x-padding
        ax.autoscale(enable=True, axis="x", tight=True)

    plt.savefig(filename)
    pass


def plot_and_save_bands_velocity_imass_v2(filename, energy, velocity, curvature, ib, nkpoints_list, labels, efermi, erange_ev, band_color='blue'):
    """
    Plots the bands
    And plots velocity and inverse effective mass for k points for which the bands lie in the given window.
    
    """
    fig, axes = plt.subplots(3, len(energy), figsize=(15, 6), sharex=True, sharey="row", gridspec_kw={
        "width_ratios": nkpoints_list,
        "wspace":0,
        "hspace":0
        }, dpi=300)
    # print(energy[0].shape)
    for k in range(len(energy)):
        x = np.linspace(0, 1, energy[k].shape[1])
        ax = axes[0, k]
        eb = energy[k][ib].T-efermi
        eb *= Ha_to_eV
        idx = np.logical_and(eb <= erange_ev[1], eb >= erange_ev[0])
        ax.plot(x, eb, label=f"E(K),{ib}", color=band_color)
        ax.axhline(erange_ev[1], color='k', linestyle="--", alpha=0.5)
        ax.axhline(erange_ev[0], color='k', linestyle="--", alpha=0.5)
        ax.axhline(0, 0, 10, color='k', linestyle=":")
        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            ax.set_ylabel(r"$E-E_F (eV)$")
            pass
        ax.set_xlabel(f"{labels[k]}")
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlim(0,1)


        ax = axes[1, k]
        # print(idx)
        # print(velocity[k].shape)
        ax.plot(x[idx], velocity[k][0, ib, idx].T, label="vx")
        ax.plot(x[idx], velocity[k][1, ib, idx].T, label="vy")
        ax.plot(x[idx], velocity[k][2, ib, idx].T, label="vz")
        # print(x[idx], velocity[k][2, ib, idx].T)
        # ax.set_ylim(-0.15, 0.15)

        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            # ax.set_ylabel(r"$v [Ha\cdot Bohr/\hbar$]")
            ax.set_ylabel(r"$v/v_0$")
        ax.set_xlabel(f"{labels[k]}")


        ax = axes[2, k]

        ax.plot(x[idx], curvature[k][0,0, ib, idx].T, label="xx")
        ax.plot(x[idx], curvature[k][1,1, ib, idx].T, label="yy")
        ax.plot(x[idx], curvature[k][2,2, ib, idx].T, label="zz")
        ax.plot(x[idx], curvature[k][0,1, ib, idx].T, label="xy")
        ax.plot(x[idx], curvature[k][0,2, ib, idx].T, label="xz")
        ax.plot(x[idx], curvature[k][1,2, ib, idx].T, label="yz")
        # ax.set_ylim(-0.15, 0.15)

        ax.axhline(0, 0, 10, color='k', linestyle=":")
        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            # ax.set_ylabel(r"$Ha\cdot Bohr^2$")
            # ax.set_ylabel(r"$m_e/m^* [Ha\cdot Bohr^2/\hbar^2$]$")
            ax.set_ylabel(r"$m_e/m^*$")
        ax.set_xlabel(f"{labels[k]}")
        axes[0, k].set_xticks([])
        axes[1, k].set_xticks([])
        axes[2, k].set_xticks([])
        pass
    for ax in axes.flat:
        # Y-axis spine
        ax.spines["left"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0.3)

        # X-axis spine
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["top"].set_linewidth(1.2)

        ax.margins(x=0)           # remove all x-padding
        ax.autoscale(enable=True, axis="x", tight=True)

    plt.savefig(filename)



def method_1(data):
    """
    using raw data
    """

    kpath_str = (
            "[0.0,0.0,0.0], [0.0,0.0,0.5]"
        )
    nkpoints = 100
    nkpoints_list = np.array([45]) * nkpoints

    kpaths_list, energies_list = extract_kpath_data(data.kpoints, data.ebands, data.atoms.cell,
                                                                    kpath_str, [100],
                                                                    fermi=data.fermi)
    

    k = 0
    x = np.linspace(0, 1, energies_list[k].shape[1])
    plt.plot(x, energies_list[k].T)
    plt.ylim(-0.03, 0.03)




    kpath_str = (
            "[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], "
                "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], "
                "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"
            )
    labels=["G-M", "M-K", "K-G", "G-A", "A-L", "L-H", "H-A", "A-L", "L-M", "M-K", "K-H"]
    nkpoints = 10
    nkpoints_list = np.array([39, 23, 45, 94, 29, 23, 45, 1, 94, 1, 94, 1]) * nkpoints

    kpaths_list, energies_list = extract_kpath_data(data.kpoints, data.ebands, data.atoms.cell,
                                                                    kpath_str, nkpoints_list,
                                                                    fermi=data.fermi)
    


    k = 3
    x = np.linspace(0, 1, energies_list[k].shape[1])
    plt.plot(x, energies_list[k].T)
    plt.xlabel(f"{labels[k]}")
    plt.legend()
    plt.ylim(-0.04, 0.04)
    

def method_2(data, equivalences, coeffs):
    """
    Using boltztrap interpolation 
    """

    # Testing
    kpath_str = "[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], " \
            "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], " \
            "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"

    kpoints, energy, velocity, curvature = extract_kpath_interpolate(data, equivalences, coeffs, kpath_str, [100])




    # Testing 2. Each k-path direction is seperated, this way the retured data will be seperated by path as well. 
    # Easier to plot when multiple paths are there
    kpath_str = ["[0.0,0.0,0.0], [0.5, 0.0, 0.0]", 
             "[0.5, 0.0, 0.0], [0.333333,0.333333,0.0]",
             "[0.333333,0.333333,0.0], [0.0,0.0,0.0]",
             "[0.0,0.0,0.0], [0.0,0.0,0.5]",
             "[0.0,0.0,0.5], [0.5,0.0,0.5]", 
             "[0.5,0.0,0.5], [0.333333,0.333333,0.5]", 
             "[0.333333,0.333333,0.5],[0.0,0.0,0.5]",
             "[0.5,0.0,0.5], [0.5, 0.0, 0.0]",
             "[0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"]


    labels=["G-M", "M-K", "K-G", "G-A", "A-L", "L-H", "H-A", "L-M", "K-H"]
    nkpoints = 1
    nkpoints_list = np.array([39, 23, 45, 94, 29, 23, 45, 94, 94]) * nkpoints

    kpoints, energy, velocity, curvature = extract_kpath_interpolate(data, equivalences, coeffs, kpath_str, nkpoints_list)

    plot_and_save_bands_velocity_imass(f"Nb3S4-bt2-bands-velocity-curvature-ib{61}.png", 
                                                    energy, velocity, curvature, 61, 
                                                    nkpoints_list, labels, data.fermi, band_color='red')
    
    plot_and_save_bands_velocity_imass(f"Nb3S4-bt2-bands-velocity-curvature-ib{62}.png", 
                                                    energy, velocity, curvature, 62, 
                                                    nkpoints_list, labels, data.fermi, band_color='green')
    
    plot_and_save_bands_velocity_imass(f"Nb3S4-bt2-bands-velocity-curvature-ib{63}.png", 
                                                    energy, velocity, curvature, 63, 
                                                    nkpoints_list, labels, data.fermi, band_color='blue')
    
    plot_and_save_bands_velocity_imass(f"Nb3S4-bt2-bands-velocity-curvature-ib{64}.png", 
                                                    energy, velocity, curvature, 64, 
                                                    nkpoints_list, labels, data.fermi, band_color='orange')

    pass

def main():

    dft_data_dir = "./PBEsol-Relaxed/"
    Efermi_DFT = 0
    niter = 20
    bt2filnam = dft_data_dir + "Nb3S4_BLZTRP_m{}.bt2".format(niter)


    data, equivalences, coeffs = compute.load_interpolation(dft_data_dir, bt2filnam, niter)
    # method_1(data)
    method_2(data, equivalences, coeffs)
    


