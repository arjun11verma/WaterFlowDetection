from gpiozero import DistanceSensor

def get_depth(echo_pin, trigger_pin):
    ultrasonic = DistanceSensor(echo=echo_pin, trigger=trigger_pin)

    for i in range(1000):
        distance = ultrasonic.distance

    distance = 0
    for i in range(100):
        distance += ultrasonic.distance
    
    return (distance / 100)