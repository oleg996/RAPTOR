extends Node3D
@export var track: NodePath 

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	position.x = get_node(track).global_position.x
	position.z = get_node(track).global_position.z
	position.y = get_node(track).global_position.y
	
	
	global_rotation.y = get_node(track).global_rotation.yd
	
