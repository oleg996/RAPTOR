extends Camera3D
@export var track: NodePath 

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	position.x = get_node(track).global_position.x-2
	position.z = get_node(track).position.z + 2.5	
