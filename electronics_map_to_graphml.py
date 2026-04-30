import argparse
from pathlib import Path
import textwrap
from xml.etree.ElementTree import Element, SubElement, ElementTree, register_namespace

import pandas as pd

# register yEd namespace
register_namespace("y", "http://www.yworks.com/xml/graphml")


def smart_wrap_label(text, width=16):
    if not isinstance(text, str):
        text = str(text)
    return '\n'.join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def is_shown(val):
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "y", "show")


# helper: create a device node (with ports & label) under given graph element
def add_device_node(parent_graph, device, ports, name_map):
    node = SubElement(parent_graph, 'node', id=device)
    data = SubElement(node, 'data', key="d0")
    shape = SubElement(data, 'y:ShapeNode')

    SubElement(shape, 'y:Geometry', width="160.0", height="120.0")
    SubElement(shape, 'y:Fill', hasColor="false", transparent="true", color="#FFFFFF")
    SubElement(shape, 'y:BorderStyle', color="#000000", type="line", width="1.0")
    # yEd sometimes exports <y:Shape> vs <y:Shape type="rectangle">; match previous style
    SubElement(shape, 'y:Shape', type="rectangle")

    device_label = device
    if device in name_map and str(name_map[device]).strip():
        wrapped_name = smart_wrap_label(name_map[device], width=16)
        device_label += f"\n{wrapped_name}"

    SubElement(shape, 'y:NodeLabel', {
        "alignment": "center",
        "autoSizePolicy": "node_width",
        "configuration": "AutoFlippingLabel",
        "fontFamily": "Times New Roman",
        "fontSize": "12",
        "fontStyle": "plain",
        "hasBackgroundColor": "false",
        "hasLineColor": "false",
        "horizontalTextPosition": "center",
        "iconTextGap": "4",
        "modelName": "internal",
        "modelPosition": "c",
        "textColor": "#000000",
        "verticalTextPosition": "bottom",
        "visible": "true",
        "wrap": "word",
        "width": "160.0",
        "height": "30.578125",
        "x": "0.0",
        "y": "44.7109375",
        "xml:space": "preserve"
    }).text = device_label

    for port in ports:
        SubElement(node, 'port', name=port)


# helper: add group node with open/closed realizers and internal graph
def add_group_node(parent_graph, group_name, member_devices, device_ports, name_map):
    group_id = f"group_{group_name}"
    group_node = SubElement(parent_graph, 'node', id=group_id)
    data = SubElement(group_node, 'data', key="d0")
    proxy = SubElement(data, 'y:ProxyAutoBoundsNode')
    realizers = SubElement(proxy, 'y:Realizers', active="0")

    # open realizer
    open_realizer = SubElement(realizers, 'y:GroupNode')
    SubElement(open_realizer, 'y:Geometry', width="696.5888888888894", height="21.666015625", x="0.0", y="0.0")
    SubElement(open_realizer, 'y:Fill', color="#F5F5F5", transparent="false")
    SubElement(open_realizer, 'y:BorderStyle', color="#FF0000", type="dashed", width="1.0")
    # label on open
    SubElement(open_realizer, 'y:NodeLabel', {
        "alignment": "center",
        "autoSizePolicy": "node_width",
        "backgroundColor": "#EBEBEB",
        "borderDistance": "0.0",
        "fontFamily": "Times New Roman",
        "fontSize": "16",
        "fontStyle": "bold",
        "hasLineColor": "false",
        "height": "21.666015625",
        "horizontalTextPosition": "center",
        "iconTextGap": "4",
        "modelName": "internal",
        "modelPosition": "t",
        "textColor": "#000000",
        "verticalTextPosition": "bottom",
        "visible": "true",
        "width": "696.5888888888894",
        "x": "0.0",
        "y": "0.0",
        "xml:space": "preserve"
    }).text = group_name
    SubElement(open_realizer, 'y:Shape', type="roundrectangle")
    SubElement(open_realizer, 'y:State', closed="false", closedHeight="50.0", closedWidth="50.0", innerGraphDisplayEnabled="false")
    SubElement(open_realizer, 'y:Insets', bottom="15", bottomF="15.0", left="15", leftF="15.0", right="15", rightF="15.0", top="15", topF="15.0")
    SubElement(open_realizer, 'y:BorderInsets', bottom="0", bottomF="0.0", left="51", leftF="50.930952380952476", right="16", rightF="15.5", top="0", topF="0.0")

    # closed realizer
    closed_realizer = SubElement(realizers, 'y:GroupNode')
    SubElement(closed_realizer, 'y:Geometry', height="50.0", width="50.0", x="0.0", y="60.0")
    SubElement(closed_realizer, 'y:Fill', color="#F5F5F5", transparent="false")
    SubElement(closed_realizer, 'y:BorderStyle', color="#FF0000", type="dashed", width="1.0")
    SubElement(closed_realizer, 'y:NodeLabel', {
        "alignment": "right",
        "autoSizePolicy": "node_width",
        "backgroundColor": "#EBEBEB",
        "borderDistance": "0.0",
        "fontFamily": "Dialog",
        "fontSize": "15",
        "fontStyle": "plain",
        "hasLineColor": "false",
        "height": "21.666015625",
        "horizontalTextPosition": "center",
        "iconTextGap": "4",
        "modelName": "internal",
        "modelPosition": "t",
        "textColor": "#000000",
        "verticalTextPosition": "bottom",
        "visible": "true",
        "width": "63.75830078125",
        "x": "-6.879150390625",
        "y": "0.0",
        "xml:space": "preserve"
    }).text = group_name
    SubElement(closed_realizer, 'y:Shape', type="roundrectangle")
    SubElement(closed_realizer, 'y:State', closed="true", closedHeight="50.0", closedWidth="50.0", innerGraphDisplayEnabled="false")
    SubElement(closed_realizer, 'y:Insets', bottom="5", bottomF="5.0", left="5", leftF="5.0", right="5", rightF="5.0", top="5", topF="5.0")
    SubElement(closed_realizer, 'y:BorderInsets', bottom="0", bottomF="0.0", left="0", leftF="0.0", right="0", rightF="0.0", top="0", topF="0.0")

    # internal graph with members
    internal_graph = SubElement(group_node, 'graph', edgedefault="directed", id=f"{group_id}:")
    for dev in member_devices:
        ports = device_ports.get(dev, set())
        add_device_node(internal_graph, dev, ports, name_map)


def build_graphml(excel_file, output_file, connection_sheet_name="Connection Sheet", device_sheet_name="Device Sheet", device_header=1):
    excel_file = Path(excel_file)
    output_file = Path(output_file)

    # load data
    df = pd.read_excel(excel_file, sheet_name=connection_sheet_name)
    device_df = pd.read_excel(excel_file, sheet_name=device_sheet_name, header=device_header)
    df.fillna('', inplace=True)
    device_df.fillna('', inplace=True)

    # name mapping
    name_map = dict(zip(device_df['Number'].astype(str), device_df['Name']))

    # determine which devices are eligible to show based on "Show?" column and group membership
    show_set = set()
    group_map = {}
    groups = {}
    for _, row in device_df.iterrows():
        dev = str(row['Number']).strip()
        group_name = str(row.get('Group', '')).strip()
        show_flag = is_shown(row.get('Show?', False))
        if not show_flag:
            continue
        show_set.add(dev)
        if group_name:
            group_map[dev] = group_name
            groups.setdefault(group_name, []).append(dev)
        else:
            group_map[dev] = ''

    # First pass: collect edges between shown devices, build device_ports, and track actual connected devices
    device_ports = {}
    connected_devices = set()
    edge_rows = []  # keep cleaned rows to iterate later

    for _, row in df.iterrows():
        s_dev = str(row.get('re', '')).strip()
        t_dev = str(row.get('Target Device Code', '')).strip()
        raw_s_port = str(row.get('Source Port', '')).strip()
        raw_t_port = str(row.get('Target Port', '')).strip()

        if s_dev not in show_set or t_dev not in show_set:
            continue  # skip if either endpoint is hidden

        # mark as connected
        connected_devices.add(s_dev)
        connected_devices.add(t_dev)

        # accumulate ports
        device_ports.setdefault(s_dev, set())
        device_ports.setdefault(t_dev, set())

        if raw_s_port:
            device_ports[s_dev].add(raw_s_port)
        elif raw_t_port:
            device_ports[s_dev].add("N/A")

        if raw_t_port:
            device_ports[t_dev].add(raw_t_port)
        elif raw_s_port:
            device_ports[t_dev].add("N/A")

        edge_rows.append({
            "s_dev": s_dev,
            "t_dev": t_dev,
            "raw_s_port": raw_s_port,
            "raw_t_port": raw_t_port
        })

    # prune to only actually connected devices
    show_set &= connected_devices
    device_ports = {dev: ports for dev, ports in device_ports.items() if dev in connected_devices}

    # rebuild groups removing empty ones
    new_groups = {}
    new_group_map = {}
    for group_name, devices in groups.items():
        filtered = [d for d in devices if d in show_set]
        if filtered:
            new_groups[group_name] = filtered
            for d in filtered:
                new_group_map[d] = group_name
    groups = new_groups
    group_map = new_group_map

    # Ensure that any shown-but-no-ports device still appears with empty set
    for dev in show_set:
        device_ports.setdefault(dev, set())

    # create graphml structure
    graphml = Element('graphml', {
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xmlns:y": "http://www.yworks.com/xml/graphml",
        "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns "
                              "http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd"
    })
    SubElement(graphml, 'key', id="d0", attrib={"for": "node", "yfiles.type": "nodegraphics"})
    SubElement(graphml, 'key', id="d1", attrib={"for": "edge", "yfiles.type": "edgegraphics"})
    graph = SubElement(graphml, 'graph', edgedefault="directed")

    # add group nodes (after pruning)
    for group_name, devices in groups.items():
        add_group_node(graph, group_name, devices, device_ports, name_map)

    # add ungrouped device nodes
    for device, ports in device_ports.items():
        if device not in show_set:
            continue
        if group_map.get(device, ''):
            continue
        add_device_node(graph, device, ports, name_map)

    # add edges with labels (only between shown + connected devices)
    for info in edge_rows:
        s_dev = info["s_dev"]
        t_dev = info["t_dev"]
        raw_s_port = info["raw_s_port"]
        raw_t_port = info["raw_t_port"]

        # create edge element
        if not raw_s_port and not raw_t_port:
            edge = SubElement(graph, 'edge', {"source": s_dev, "target": t_dev})
        else:
            s_port = raw_s_port if raw_s_port else "N/A"
            t_port = raw_t_port if raw_t_port else "N/A"
            edge_attrs = {
                "source": s_dev,
                "target": t_dev,
                "sourceport": s_port,
                "targetport": t_port
            }
            edge = SubElement(graph, 'edge', edge_attrs)

        # edge graphics
        edata = SubElement(edge, 'data', key="d1")
        polyline = SubElement(edata, 'y:PolyLineEdge')
        SubElement(polyline, 'y:LineStyle', color="#000000", type="line", width="1.0")
        SubElement(polyline, 'y:Arrows', source="none", target="standard")

        if raw_s_port or raw_t_port:
            s_port = raw_s_port if raw_s_port else "N/A"
            t_port = raw_t_port if raw_t_port else "N/A"
            edge_label_text = f"{s_port} → {t_port}"

            label_attrs = {
                "alignment": "center",
                "distance": "-2.0",
                "fontFamily": "Times New Roman",
                "fontSize": "6",
                "fontStyle": "plain",
                "hasBackgroundColor": "false",
                "hasLineColor": "false",
                "horizontalTextPosition": "center",
                "iconTextGap": "4",
                "modelName": "custom",
                "preferredPlacement": "anywhere",
                "ratio": "0.5",
                "textColor": "#0000FF",
                "verticalTextPosition": "bottom",
                "visible": "true",
                "width": "30.8388671875",
                "height": "10.64453125",
                "xml:space": "preserve"
            }
            edge_label_elem = SubElement(polyline, 'y:EdgeLabel', label_attrs)
            edge_label_elem.text = edge_label_text

            # LabelModel with RotatedDiscreteEdgeLabelModel
            label_model = SubElement(edge_label_elem, 'y:LabelModel')
            SubElement(label_model, 'y:RotatedDiscreteEdgeLabelModel', {
                "angle": "0.0",
                "autoRotationEnabled": "true",
                "candidateMask": "18",
                "distance": "-2.0",
                "positionRelativeToSegment": "false"
            })

            # ModelParameter
            model_param = SubElement(edge_label_elem, 'y:ModelParameter')
            SubElement(model_param, 'y:RotatedDiscreteEdgeLabelModelParameter', {
                "position": "head"
            })

            # PreferredPlacementDescriptor
            SubElement(edge_label_elem, 'y:PreferredPlacementDescriptor', {
                "angle": "0.0",
                "angleOffsetOnRightSide": "0",
                "angleReference": "absolute",
                "angleRotationOnRightSide": "co",
                "distance": "-1.0",
                "frozen": "true",
                "placement": "anywhere",
                "side": "anywhere",
                "sideReference": "relative_to_edge_flow"
            })

    # save to GraphML file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ElementTree(graphml).write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Exported: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a yEd GraphML electronics connection diagram from an Excel file."
    )
    parser.add_argument(
        "excel_file",
        help="Path to the input Excel file, for example: 'LiCs Electronics Map.xlsx'"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output .graphml file. If omitted, uses the Excel file name with .graphml extension."
    )
    parser.add_argument(
        "--connection-sheet",
        default="Connection Sheet",
        help="Name of the connection sheet. Default: 'Connection Sheet'"
    )
    parser.add_argument(
        "--device-sheet",
        default="Device Sheet",
        help="Name of the device sheet. Default: 'Device Sheet'"
    )
    parser.add_argument(
        "--device-header",
        type=int,
        default=1,
        help="Header row index for the device sheet, using pandas zero-based indexing. Default: 1"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    excel_path = Path(args.excel_file)
    output_path = Path(args.output) if args.output else excel_path.with_suffix(".graphml")

    build_graphml(
        excel_file=excel_path,
        output_file=output_path,
        connection_sheet_name=args.connection_sheet,
        device_sheet_name=args.device_sheet,
        device_header=args.device_header,
    )


if __name__ == "__main__":
    main()
