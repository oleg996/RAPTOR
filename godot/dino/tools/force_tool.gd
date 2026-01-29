extends Node2D

var pressed = false
var to_grab = null
var grabpos = Vector2(0,0)
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	position = get_global_mouse_position()
	
	if pressed:
		var lpos = grabpos.rotated(to_grab.rotation)+to_grab.position
		
		to_grab.apply_force((position-lpos).normalized()*5000)

	
func grab():
	var tab = $Area2D.get_overlapping_bodies()
	if tab.size() != 0:
		to_grab = $Area2D.get_overlapping_bodies()[0]
		grabpos = (position - to_grab.position).rotated(to_grab.rotation * -1)
	else:
		pressed = false
	

func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_mask == 2:
			pressed = true
			grab()
			print("grabbed")
		if event.button_mask == 0:
			pressed = false
