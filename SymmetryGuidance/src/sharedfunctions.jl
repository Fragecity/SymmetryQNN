function is_inCircle((x, y)::Tuple, center::Tuple, radius::AbstractFloat)
	distance_square = (x - center[1])^2 + (y - center[2])^2
	return distance_square <= radius^2
end

function label_double_circle_data((x, y)::Tuple, radius_inner::AbstractFloat, radius_outer::AbstractFloat, center::Tuple)
	label_inner = is_inCircle((x, y), center, radius_inner) ? 0 : 1
	label_outer = is_inCircle((x, y), center, radius_outer) ? 0 : 1
	return label_inner + label_outer
end

function label_circ_data((x, y)::Tuple, radius::AbstractFloat, center::Tuple)
    is_inCircle((x, y), center, radius) ? 0 : 1
end