from air_conditioning_power_generator import AirConditioningPowerGenerator


def main():
    """
    Main function to initialize fuzzy logic POC.

    This function set up our Air Conditioning Generator with fuzzy logic inside.
    """

    temperature_inside = 30
    temperature_outside = 40
    number_of_people = 10

    air_conditioning_power_generator = AirConditioningPowerGenerator(temperature_inside, temperature_outside, number_of_people)
    air_conditioning_power_generator.generate()

if __name__ == '__main__':
    main()