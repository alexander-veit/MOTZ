
import numpy
import matplotlib.pyplot as plt
import meshplex
import uuid
import math

def MOTZ(
    mesh,
    N_test=None,
    N_dof=None,
    detailed_output=True,
    save_plots=False
):

    if N_test is None:
        nodes = numpy.arange(len(mesh.node_coords))
        N_test = nodes[mesh.is_boundary_node]
        N_dof = nodes[~mesh.is_boundary_node]

    mesh.create_edges()
    N_test = list(N_test)
    N_dof = list(N_dof)

    shortUuid = str(uuid.uuid4())[:4]

    print("MOTZ started")

    MOTZ_result = "certified"
    MOTZ_trans = True
    MOTZ_angle = True

    num_outer_iterations = 0
    next_node_found = True

    while len(N_dof) != 0 and MOTZ_trans and num_outer_iterations < 200: 

        next_node_found = False
        if detailed_output:
            sp = 'images/MOTZ_' + shortUuid + '_' + str(num_outer_iterations) + '.eps'  if save_plots else None

            show(mesh=mesh, 
                show_coedges=False, 
                boundary_edge_color=(0.17,0.51,1.0), 
                show_node_numbers=False, 
                orange_nodes=N_test,
                red_nodes=N_dof,
                show_axes=False,
                save_plot_as=sp
            )
            print("Outer Iteration ", num_outer_iterations)

        for z in N_dof:
            if detailed_output:
                print("Inner Iteration - Testing node ", z)
            neighbors = get_neighbor_nodes(z,mesh)
            z_prime_found = False
            for z_prime in neighbors:
                if z_prime in N_test and get_transmission_degree(z_prime, mesh, N_test, N_dof) == 1:
                    if get_transmission_angle(z,z_prime,mesh) > math.pi:
                        print("Warning: Acute angle condition violated for nodes", z, z_prime)
                        MOTZ_angle = False
                    z_prime_found = True
            
            if z_prime_found:
                N_test.append(z)
                N_dof.remove(z)
                next_node_found = True
                break

        if not next_node_found:
            MOTZ_trans = False

        num_outer_iterations = num_outer_iterations + 1

    if detailed_output:
        sp = 'images/MOTZ_' + shortUuid + '_' + str(num_outer_iterations) + '.eps'  if save_plots else None
        show(mesh=mesh, 
            show_coedges=False, 
            boundary_edge_color=(0.17,0.51,1.0), 
            show_node_numbers=False, 
            orange_nodes=N_test,
            red_nodes=N_dof,
            show_axes=False,
            save_plot_as=sp
        )

    if(MOTZ_angle == False or MOTZ_trans == False):
        MOTZ_result = "critical" 

    print("MOTZ complete.")
    print("MOTZ_result: ", MOTZ_result)
    print("MOTZ_angle: ", MOTZ_angle)
    print("MOTZ_trans: ", MOTZ_trans)

    return N_dof


def MOTZ_flip(
    X,
    mesh,
    N_dof=None,
    detailed_output=True,
    save_plots=False
):

    # contains triangle id's where edge flipping makes sense
    potential_flips = []
    cells = mesh.cells['nodes']

    for z in N_dof:

        neighbors = get_neighbor_nodes(z,mesh)
        neighbors = list_diff(neighbors, N_dof) # remove N_dof

        if len(neighbors) == 2:
            # Get the common node of the neighbors
            neighbors_node_1 = get_neighbor_nodes(neighbors[0],mesh)
            neighbors_node_1 = list_diff(neighbors_node_1, N_dof) # remove N_dof
            neighbors_node_2 = get_neighbor_nodes(neighbors[1],mesh)
            neighbors_node_2 = list_diff(neighbors_node_2, N_dof) # remove N_dof

            common_node = list_intersection(neighbors_node_1, neighbors_node_2)
            if len(common_node) != 1:
                continue
            common_node = common_node[0]

            # The triangles are 
            # [z, neighbors[0], neighbors[1]] and
            # [neighbors[0], neighbors[1], common_node]
            triangle_1 = [z, neighbors[0], neighbors[1]]
            triangle_2 = [neighbors[0], neighbors[1], common_node]

            cell_ids = []
            for i in range(len(cells)):
                cur_triangle = cells[i].tolist()
                if( len(list_intersection(cur_triangle, triangle_1)) == 3 or
                    len(list_intersection(cur_triangle, triangle_2)) == 3 ):
                    cell_ids.append(i)
            if len(cell_ids) == 2:
                potential_flips.append(cell_ids)

    if(len(potential_flips)>0):
        flip_tr = potential_flips[0]
        flip_edge(cells, flip_tr[0], flip_tr[1])
        mesh = meshplex.MeshTri(X, cells)

        if detailed_output:
            sp = 'images/improve_mesh.eps' if save_plots else None
            show(mesh=mesh, 
            show_coedges=False, 
            boundary_edge_color=(0.17,0.51,1.0), 
            show_node_numbers=False, 
            orange_nodes=None,
            red_nodes=None,
            show_axes=False,
            save_plot_as=sp
        )
        return mesh
    
    return

            
def list_intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def list_diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def lengthSquare(X, Y): 
    xDiff = X[0] - Y[0] 
    yDiff = X[1] - Y[1] 
    return xDiff * xDiff + yDiff * yDiff

def get_transmission_angle(
    z,
    z_prime,
    mesh
):
    nodes = mesh.node_coords

    neighbors_z = get_neighbor_nodes(z,mesh)
    neighbors_zp = get_neighbor_nodes(z_prime,mesh)
    neighbors_common = list_intersection(neighbors_zp,neighbors_z)
    if len(neighbors_common) != 2:
        raise ValueError("WARNING: Could not detect exactly 2 neighboring triangles while calculating the acute angle condition")

    triangle_node_1 = neighbors_common[0]
    A = nodes[z]
    B = nodes[z_prime]
    C = nodes[triangle_node_1]

    c2 = lengthSquare(A, B)
    b2 = lengthSquare(A, C)
    a2 = lengthSquare(B, C)

    gamma_1 = math.acos((a2 + b2 - c2) / 
                         (2 * math.sqrt(a2) * math.sqrt(b2))); 

    triangle_node_2 = neighbors_common[1]
    C = nodes[triangle_node_2]

    c2 = lengthSquare(A, B)
    b2 = lengthSquare(A, C)
    a2 = lengthSquare(B, C)

    gamma_2 = math.acos((a2 + b2 - c2) / 
                         (2 * math.sqrt(a2) * math.sqrt(b2))); 

    return gamma_1 + gamma_2


def get_transmission_degree(
    node_id,
    mesh,
    N_test,
    N_dof
):
    if node_id not in N_test:
        raise ValueError("WARNING: node_id not in N_test in get_transmission_degree")

    result_set = []
    neighbors = get_neighbor_nodes(node_id,mesh)

    for neighbor in neighbors:
        if neighbor in N_dof:
            result_set.append(neighbor)

    return len(result_set)

def get_neighbor_nodes(
    node_id,
    mesh
):
    neighbors = []
    edgeArr = mesh.edges['nodes']

    for edge in edgeArr:
        if edge[0] == node_id:
            neighbors.append(edge[1])
        elif edge[1] == node_id:
            neighbors.append(edge[0])

    return neighbors

def show(mesh, *args, **kwargs):
    """Show the mesh (see plot()).
    """
    plot(mesh, *args, **kwargs)
    plt.show()
    plt.close()
    return

def plot(
    mesh,
    show_coedges=True,
    control_volume_centroid_color=None,
    mesh_color="k",
    nondelaunay_edge_color=None,
    boundary_edge_color=None,
    comesh_color=(0.8, 0.8, 0.8),
    show_axes=True,
    cell_quality_coloring=None,
    show_node_numbers=False,
    show_cell_numbers=False,
    cell_mask=None,
    orange_nodes=None,
    red_nodes=None,
    save_plot_as=None
):
    """Show the mesh using matplotlib.
    """
    # Importing matplotlib takes a while, so don't do that at the header.
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure()
    ax = fig.gca()
    plt.axis("equal")
    if not show_axes:
        ax.set_axis_off()

    xmin = numpy.amin(mesh.node_coords[:, 0])
    xmax = numpy.amax(mesh.node_coords[:, 0])
    ymin = numpy.amin(mesh.node_coords[:, 1])
    ymax = numpy.amax(mesh.node_coords[:, 1])

    width = xmax - xmin
    xmin -= 0.1 * width
    xmax += 0.1 * width

    height = ymax - ymin
    ymin -= 0.1 * height
    ymax += 0.1 * height

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if show_node_numbers:
        for i, x in enumerate(mesh.node_coords):
            plt.text(
                x[0],
                x[1],
                str(i),
                bbox=dict(facecolor="w", alpha=0.7),
                horizontalalignment="center",
                verticalalignment="center",
            )

    if show_cell_numbers:
        for i, x in enumerate(mesh.cell_centroids):
            plt.text(
                x[0],
                x[1],
                str(i),
                bbox=dict(facecolor="r", alpha=0.5),
                horizontalalignment="center",
                verticalalignment="center",
            )

    # coloring
    if cell_quality_coloring:
        cmap, cmin, cmax, show_colorbar = cell_quality_coloring
        plt.tripcolor(
            mesh.node_coords[:, 0],
            mesh.node_coords[:, 1],
            mesh.cells["nodes"],
            mesh.cell_quality,
            shading="flat",
            cmap=cmap,
            vmin=cmin,
            vmax=cmax,
        )
        if show_colorbar:
            plt.colorbar()
    
    if not orange_nodes is None:
        for i, x in enumerate(orange_nodes):
            plt.plot(mesh.node_coords[x][0],mesh.node_coords[x][1], marker='o', markersize=10, color="orange")
    
    if not red_nodes is None:
        for i, x in enumerate(red_nodes):
            plt.plot(mesh.node_coords[x][0],mesh.node_coords[x][1], marker='o', markersize=10, color="red")

    if mesh.edges is None:
        mesh.create_edges()

    # Get edges, cut off z-component.
    e = mesh.node_coords[mesh.edges["nodes"]][:, :, :2]

    if nondelaunay_edge_color is None:
        line_segments0 = LineCollection(e, color=mesh_color)
        ax.add_collection(line_segments0)
    else:
        # Plot regular edges, mark those with negative ce-ratio red.
        ce_ratios = mesh.ce_ratios_per_interior_edge
        pos = ce_ratios >= 0

        is_pos = numpy.zeros(len(mesh.edges["nodes"]), dtype=bool)
        is_pos[mesh._edge_to_edge_gid[2][pos]] = True

        # Mark Delaunay-conforming boundary edges
        is_pos_boundary = mesh.ce_ratios[mesh.is_boundary_edge] >= 0
        is_pos[mesh._edge_to_edge_gid[1][is_pos_boundary]] = True

        line_segments0 = LineCollection(e[is_pos], color=mesh_color)
        ax.add_collection(line_segments0)

        line_segments1 = LineCollection(e[~is_pos], color=nondelaunay_edge_color)
        ax.add_collection(line_segments1)

    if show_coedges:
        # Connect all cell circumcenters with the edge midpoints
        cc = mesh.cell_circumcenters

        edge_midpoints = 0.5 * (
            mesh.node_coords[mesh.edges["nodes"][:, 0]]
            + mesh.node_coords[mesh.edges["nodes"][:, 1]]
        )

        # Plot connection of the circumcenter to the midpoint of all three
        # axes.
        a = numpy.stack(
            [cc[:, :2], edge_midpoints[mesh.cells["edges"][:, 0], :2]], axis=1
        )
        b = numpy.stack(
            [cc[:, :2], edge_midpoints[mesh.cells["edges"][:, 1], :2]], axis=1
        )
        c = numpy.stack(
            [cc[:, :2], edge_midpoints[mesh.cells["edges"][:, 2], :2]], axis=1
        )

        line_segments = LineCollection(
            numpy.concatenate([a, b, c]), color=comesh_color
        )
        ax.add_collection(line_segments)

    if boundary_edge_color:
        e = mesh.node_coords[mesh.edges["nodes"][mesh.is_boundary_edge_individual]][
            :, :, :2
        ]
        line_segments1 = LineCollection(e, color=boundary_edge_color)
        ax.add_collection(line_segments1)

    if control_volume_centroid_color is not None:
        centroids = mesh.get_control_volume_centroids(cell_mask=cell_mask)
        ax.plot(
            centroids[:, 0],
            centroids[:, 1],
            linestyle="",
            marker=".",
            color=control_volume_centroid_color,
        )
        for k, centroid in enumerate(centroids):
            plt.text(
                centroid[0],
                centroid[1],
                str(k),
                bbox=dict(facecolor=control_volume_centroid_color, alpha=0.7),
                horizontalalignment="center",
                verticalalignment="center",
            )

    if save_plot_as is not None:
        plt.savefig(save_plot_as, format='eps', bbox_inches = 'tight', pad_inches = 0)

    return fig


def flip_edge(cells, cell_id_1, cell_id_2):
    """
    Flip the edge of the cells with id's cell_id_1 and cell_id_2
    """

    # Find common edge
    cell_1 = cells[cell_id_1]
    cell_2 = cells[cell_id_2]

    common_edge = numpy.intersect1d(cell_1, cell_2)
    if len(common_edge) != 2:
        print("Triangels have no common edge.")
        return

    cell_1_3rd_vertex = numpy.setdiff1d(cell_1, common_edge)
    cell_2_3rd_vertex = numpy.setdiff1d(cell_2, common_edge)

    cell_1_mod = numpy.array([cell_1_3rd_vertex, common_edge[0], cell_2_3rd_vertex])
    cell_2_mod = numpy.array([cell_1_3rd_vertex, common_edge[1], cell_2_3rd_vertex])

    cells[cell_id_1] = cell_1_mod
    cells[cell_id_2] = cell_2_mod


    return