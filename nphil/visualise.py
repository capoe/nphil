from . import maths
import nphil._nphil as _nphil
import numpy as np
import scipy.stats
import json

def represent_graph_2d_linear(
        fgraph, 
        root_weights, 
        alpha, 
        weight_fct=lambda f: np.abs(f.cov*f.confidence)):
    # POSITION NODES
    scale = 10
    dy_root = scale*5./len(fgraph.map_generations[0])
    y_max = scale*5.
    x_offset = 0.0
    x_root = scale*0.5
    qc_scale = scale*0.0

    unary_offset = [ None, 0.55*x_root ]
    unary_offset_y = [ None, 0. ]
    unary_op_offset = { 
        "e": [ 0.0, 1.0 ],
        "l": [ -1.0, 0.5 ],
        "|": [ +1.0, 0.5 ],
        "s": [ -1.0, -0.5 ],
        "r": [ +1.0, -0.5 ],
        "2": [ 0.0, -1.0 ]
    }
    binary_offset = [ None, 1.55*x_root, 3*x_root ]
    binary_offset_y = [ None, 0., 0. ]
    root_off = 0.0
    w_avg = np.average([ w for r,w in root_weights.iteritems() ])

    dy_map = {}
    for gen in fgraph.generations:
        nodes = fgraph.map_generations[gen]
        print("Positioning generation", gen)
        for idx, node in enumerate(nodes):
            if gen == 0:
                w = root_weights[node.expr]
                node.x = x_offset
                dy = w/w_avg*dy_root
                dy_map[node.expr] = dy
                node.y = root_off + 0.5*dy
                y_max = node.y
                root_off += dy
                print("x=%1.2f y=%1.2f %s" % (node.x, node.y, node.expr))
            elif len(node.parents) == 1:
                # Unary case
                par = node.parents[0]
                node.x = unary_offset[gen] + qc_scale*weight_fct(node) + x_offset + 0.1*unary_offset[gen]*unary_op_offset[node.fnode_.op_tag][0]
                node.y = unary_offset_y[gen] + 0.0*dy_root + par.y + 0.2*dy_map[par.expr]*unary_op_offset[node.fnode_.op_tag][1]
            elif len(node.parents) == 2:
                # Binary case
                p1 = node.parents[0]
                p2 = node.parents[1]
                y_parents = sorted([ p.y for p in node.parents ])
                dy = y_parents[1]-y_parents[0]
                node.x = 0.05*(p2.y-p1.y) + binary_offset[gen] + qc_scale*(
                    weight_fct(node)) + x_offset
                w1 = w2 = 0.5
                node.y = binary_offset_y[gen] + w1*p1.y+w2*p2.y + qc_scale*(
                    weight_fct(node)**2)
    # LINKS BETWEEN NODES
    def connect_straight(f1, f2):
        x1 = f1.x
        y1 = f1.y
        x2 = f2.x
        y2 = f2.y
        w = weight_fct(f2)
        return [ [x1,y1,w], [x2,y2,w] ]
    def connect_tanh(f0, f1, f2, samples=30, alpha=alpha):
        x1 = f1.x
        y1 = f1.y
        x2 = f2.x
        y2 = f2.y
        w = weight_fct(f2)
        coords = []
        for i in range(samples):
            xi = x1 + float(i)/(samples-1)*(x2-x1)
            yi = y1 + (y2-y1)*0.5*(1 + np.tanh(alpha*(xi - 0.5*(x1+x2))))
            coords.append([xi, yi, w])
        return coords
    def connect_arc(f0, f1, f2, samples=30):
        x0 = f0.x
        y0 = f0.y
        x1 = f1.x
        y1 = f1.y
        x2 = f2.x
        y2 = f2.y
        w = weight_fct(f2)
        r1 = ((x1-x0)**2+(y1-y0)**2)**0.5
        r2 = ((x2-x0)**2+(y2-y0)**2)**0.5
        phi1 = np.arctan2(y1-y0, x1-x0)
        phi2 = np.arctan2(y2-y0, x2-x0)
        if phi1 < 0.: phi1 = 2*np.pi + phi1
        if phi2 < 0.: phi2 = 2*np.pi + phi2
        phi_start = phi1
        dphi = phi2-phi1
        if dphi >= np.pi:
            dphi = 2*np.pi - dphi
            phi_end = phi_start-dphi
        elif dphi <= -np.pi:
            dphi = 2*np.pi + dphi
            phi_end = phi_start+dphi
        else:
            phi_end = phi_start + dphi
        coords = []
        for i in range(samples):
            phi_i = phi_start + float(i)/(samples-1)*(phi_end-phi_start)
            rad_i = r1 + float(i)/(samples-1)*(r2-r1)
            x_i = x0 + rad_i*np.cos(phi_i)
            y_i = y0 + rad_i*np.sin(phi_i)
            coords.append([x_i, y_i, w])
        return coords
    curves = []
    curve_info = []
    for fnode in fgraph.fnodes:
        if len(fnode.parents) == 1:
            curve_info.append({ "target": fnode.expr, "source": fnode.parents[0].expr })
            curves.append(connect_tanh(None, fnode.parents[0], fnode, alpha=3*alpha))
        elif len(fnode.parents) == 2:
            curve_info.append({ "target": fnode.expr, "source": fnode.parents[0].expr })
            curves.append(connect_tanh(fnode.parents[0], fnode.parents[1], fnode))
            curve_info.append({ "target": fnode.expr, "source": fnode.parents[1].expr })
            curves.append(connect_tanh(fnode.parents[1], fnode.parents[0], fnode))
        else: pass
    # Sort curves so important ones are in the foreground
    order = np.argsort([ c[0][-1] for c in curves ])
    curves = [ curves[_] for _ in order ]
    curve_info = [ curve_info[_] for _ in order]
    return fgraph, curves, curve_info

def represent_graph_2d(fgraph):
    # POSITION NODES
    dphi_root = 2*np.pi/len(fgraph.map_generations[0])
    radius_offset = 0.0
    radius_root = 1.0
    radius_scale = 2.5
    for gen in fgraph.generations:
        nodes = fgraph.map_generations[gen]
        print("Positioning generation", gen)
        for idx, node in enumerate(nodes):
            if gen == 0:
                node.radius = radius_root + radius_offset
                node.phi = idx*dphi_root
                print("r=%1.2f phi=%1.2f %s" % (node.radius, node.phi, node.expr))
            elif len(node.parents) == 1:
                # Unary case
                par = node.parents[0]
                node.radius = (1.+gen-0.3)**2*radius_root + radius_scale*(
                    np.abs(node.cov*node.confidence))*radius_root + radius_offset
                node.phi = par.phi + (
                    np.abs(node.cov*node.confidence))*dphi_root/node.radius
            elif len(node.parents) == 2:
                # Binary case
                p1 = node.parents[0]
                p2 = node.parents[1]
                phi_parents = sorted([ p.phi for p in node.parents ])
                dphi = phi_parents[1]-phi_parents[0]
                if dphi <= np.pi:
                    node.phi = phi_parents[0] + 0.5*dphi
                else:
                    node.phi = (phi_parents[1] + 0.5*(2*np.pi - dphi)) % (2*np.pi)
                node.radius = (1.+gen+(0.2 if gen < 2 else 0))**2*radius_root + radius_scale*(
                    np.abs(node.cov*node.confidence))*radius_root + radius_offset
                node.phi = node.phi + (
                    np.abs(node.cov*node.confidence))*dphi_root/node.radius
    # LINKS BETWEEN NODES
    def connect_straight(f1, f2):
        x1 = f1.radius*np.cos(f1.phi)
        y1 = f1.radius*np.sin(f1.phi)
        x2 = f2.radius*np.cos(f2.phi)
        y2 = f2.radius*np.sin(f2.phi)
        #w = np.abs(f2.cov*f2.confidence)
        w = f2.rank
        return [ [x1,y1,w], [x2,y2,w] ]
    def connect_arc(f0, f1, f2, samples=15):
        x0 = f0.radius*np.cos(f0.phi)
        y0 = f0.radius*np.sin(f0.phi)
        x1 = f1.radius*np.cos(f1.phi)
        y1 = f1.radius*np.sin(f1.phi)
        x2 = f2.radius*np.cos(f2.phi)
        y2 = f2.radius*np.sin(f2.phi)
        #w = np.abs(f2.cov*f2.confidence)
        w = f2.rank
        r1 = ((x1-x0)**2+(y1-y0)**2)**0.5
        r2 = ((x2-x0)**2+(y2-y0)**2)**0.5
        phi1 = np.arctan2(y1-y0, x1-x0)
        phi2 = np.arctan2(y2-y0, x2-x0)
        if phi1 < 0.: phi1 = 2*np.pi + phi1
        if phi2 < 0.: phi2 = 2*np.pi + phi2
        phi_start = phi1
        dphi = phi2-phi1
        if dphi >= np.pi:
            dphi = 2*np.pi - dphi
            phi_end = phi_start-dphi
        elif dphi <= -np.pi:
            dphi = 2*np.pi + dphi
            phi_end = phi_start+dphi
        else:
            phi_end = phi_start + dphi
        coords = []
        for i in range(samples):
            phi_i = phi_start + float(i)/(samples-1)*(phi_end-phi_start)
            rad_i = r1 + float(i)/(samples-1)*(r2-r1)
            x_i = x0 + rad_i*np.cos(phi_i)
            y_i = y0 + rad_i*np.sin(phi_i)
            coords.append([x_i, y_i, w])
        return coords
    curves = []
    curve_info = []
    for fnode in fgraph.fnodes:
        if len(fnode.parents) == 1:
            curve_info.append({ "target": fnode.expr, "source": fnode.parents[0].expr })
            curves.append(connect_straight(fnode.parents[0], fnode))
        elif len(fnode.parents) == 2:
            curve_info.append({ "target": fnode.expr, "source": fnode.parents[0].expr })
            curves.append(connect_arc(fnode.parents[0], fnode.parents[1], fnode))
            curve_info.append({ "target": fnode.expr, "source": fnode.parents[1].expr })
            curves.append(connect_arc(fnode.parents[1], fnode.parents[0], fnode))
        else: pass
    # Sort curves so important ones are in the foreground
    order = np.argsort([ c[0][-1] for c in curves ])
    curves = [ curves[_] for _ in order ]
    curve_info = [ curve_info[_] for _ in order]
    return fgraph, curves, curve_info

