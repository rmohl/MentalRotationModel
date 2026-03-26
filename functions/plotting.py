import plotly.graph_objects as go

def add_point(fig, coords, name, colour='green'):
        
    fig.add_trace(*create_data_from_point(coords, name, colour))                # create_data returns a list, the * unpacks it

def add_landmarks(fig, landmarks, name, colour='orange'):

    showlegend = True

    for landmark in landmarks:
        fig.add_trace(go.Scatter3d(
                    x=[landmark[0]],
                    y=[landmark[1]],
                    z=[landmark[2]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colour
                    ),
                    name=name,
                    legendgroup=name,
                    showlegend=showlegend
        ))

        showlegend = False              # false after first loop!

def add_axis(fig, axis, colour='green', scale=1):

    fig.add_trace(*create_data_from_axis(axis, colour, scale))          # create_data returns a list, the * unpacks it

def add_object(fig, object, name, colour='red', row=None, col=None):

    vector_start = object.start_coords.copy()
    vector_end = object.start_coords.copy()
    showlegend = True

    for geon in object.geons:
        direction_vector = geon.direction * geon.length
        vector_start = vector_end.copy()
        vector_end += direction_vector
        
        if row == None or col == None:
            fig.add_trace(go.Scatter3d(
                x=[vector_start[0], vector_end[0]],
                y=[vector_start[1], vector_end[1]],
                z=[vector_start[2], vector_end[2]],
                mode='lines',
                line=dict(color=colour, width=5),
                name=name,
                legendgroup=name,
                showlegend=showlegend
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[vector_start[0], vector_end[0]],
                y=[vector_start[1], vector_end[1]],
                z=[vector_start[2], vector_end[2]],
                mode='lines',
                line=dict(color=colour, width=5),
                name=name,
                legendgroup=name,
                showlegend=showlegend
                ),
            row=row,
            col=col
            )

        showlegend=False            # false after first loop!

def create_data_from_point(coords, name, colour='green'):

    return [go.Scatter3d(
        x=[coords[0]],
        y=[coords[1]],
        z=[coords[2]],
        mode='markers',
        marker=dict(
            size=5,
            color=colour
        ),
        name=name,
    )]

def create_data_from_landmarks(landmarks, name, colour='orange'):

    data = []
    showlegend = True

    for landmark in landmarks:
        data.append(go.Scatter3d(
                    x=[landmark[0]],
                    y=[landmark[1]],
                    z=[landmark[2]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colour
                    ),
                    name=name,
                    legendgroup=name,
                    showlegend=showlegend
        ))

        showlegend = False              # false after first loop!

    return data

def create_data_from_axis(axis, colour='green', scale=1):
    axis_start = axis * -scale
    axis_end = axis * scale

    return [go.Scatter3d(
        x=[axis_start[0], axis_end[0]],
        y=[axis_start[1], axis_end[1]],
        z=[axis_start[2], axis_end[2]],
        mode='lines',
        line=dict(color=colour, width=5, dash='dot'),
        name="Axis of Rotation"
    )]

def create_data_from_object(object, name, colour='blue'):

    data = []
    showlegend = True

    vector_start = object.start_coords.copy()
    vector_end = object.start_coords.copy()

    for geon in object.geons:
        direction_vector = geon.direction * geon.length
        vector_start = vector_end.copy()
        vector_end += direction_vector
        
        data.append(go.Scatter3d(
            x=[vector_start[0], vector_end[0]],
            y=[vector_start[1], vector_end[1]],
            z=[vector_start[2], vector_end[2]],
            mode='lines',
            line=dict(color=colour, width=5),
            name=name,
            legendgroup=name,
            showlegend=showlegend
        ))

        showlegend=False            # false after first loop!

    return data

def add_frame(frame_array, object, object_name, landmark_name=None, axis_of_rotation=None, object_colour='blue', landmark_colour='purple', axis_colour='green', axis_scale=1):

    data = create_data_from_object(object, object_name, object_colour)

    if landmark_name is not None:
        data = data + create_data_from_landmarks(object.get_landmark_endpoints(), landmark_name, landmark_colour)
    if axis_of_rotation is not None:
        data = data + create_data_from_axis(axis_of_rotation, axis_colour, axis_scale)

    traces = [i for i in range(len(data))]
    name = f'frame{len(frame_array)}'

    frame_array.append(go.Frame(data=data, traces=traces, name=name))

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
            }
