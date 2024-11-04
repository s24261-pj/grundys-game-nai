import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class AirConditioningPowerGenerator:
    def __init__(self, temperature_inside, temperature_outside, number_of_people):
        """
        Initializes the air conditioning power generator based on indoor temperature, outdoor temperature,
        and the number of people in the room.
        """
        self.temperature_inside_input = temperature_inside
        self.temperature_outside_input = temperature_outside
        self.number_of_people_input = number_of_people

        self.temp_inside = self.__init_temp_inside_universe()
        self.temp_outside = self.__init_temp_outside_universe()
        self.num_people = self.__init_num_people_universe()

        self.cooling_power = self.__init_cooling_power_universe()

    def generate(self):
        """
        Generates and computes the cooling power needed based on the input conditions.
        Uses a fuzzy rule set to determine the output cooling power.
        """
        rules = self.__generate_rules()

        climate_control = ctrl.ControlSystem(rules)
        climate_simulation = ctrl.ControlSystemSimulation(climate_control)

        climate_simulation.input['Temperature inside'] = self.temperature_inside_input
        climate_simulation.input['Temperature outside'] = self.temperature_outside_input
        climate_simulation.input['Number of people'] = self.number_of_people_input

        climate_simulation.compute()
        print(f"Cooling power: {climate_simulation.output['Cooling power']}%")

    @staticmethod
    def __init_temp_inside_universe():
        """
        Initializes the fuzzy universe for indoor temperature with categories:
        'cool', 'comfort', 'hot', and 'very hot'.
        """
        temp_inside = ctrl.Antecedent(np.arange(10, 40, 1), 'Temperature inside')

        temp_inside['cool'] = fuzz.trimf(temp_inside.universe, [13, 18, 23])
        temp_inside['comfort'] = fuzz.trimf(temp_inside.universe, [20, 24, 28])
        temp_inside['hot'] = fuzz.trimf(temp_inside.universe, [25, 30, 35])
        temp_inside['very hot'] = fuzz.trimf(temp_inside.universe, [32, 37, 40])

        return temp_inside

    @staticmethod
    def __init_temp_outside_universe():
        """
        Initializes the fuzzy universe for outdoor temperature with categories:
        'very cold', 'cold', 'moderate', 'warm', and 'very warm'.
        """
        temp_outside = ctrl.Antecedent(np.arange(-5, 45, 1), 'Temperature outside')

        temp_outside['very cold'] = fuzz.trimf(temp_outside.universe, [-5, -5, 5])
        temp_outside['cold'] = fuzz.trimf(temp_outside.universe, [0, 10, 15])
        temp_outside['moderate'] = fuzz.trimf(temp_outside.universe, [12, 20, 28])
        temp_outside['warm'] = fuzz.trimf(temp_outside.universe, [25, 30, 35])
        temp_outside['very warm'] = fuzz.trimf(temp_outside.universe, [33, 40, 45])

        return temp_outside

    @staticmethod
    def __init_num_people_universe():
        """
        Initializes the fuzzy universe for the number of people in the room with categories:
        'empty', 'few', 'medium', 'many', and 'full'.
        """
        num_people = ctrl.Antecedent(np.arange(0, 15, 1), 'Number of people')

        num_people['empty'] = fuzz.trimf(num_people.universe, [0, 0, 0])
        num_people['few'] = fuzz.trimf(num_people.universe, [0, 2, 4])
        num_people['medium'] = fuzz.trimf(num_people.universe, [3, 6, 9])
        num_people['many'] = fuzz.trimf(num_people.universe, [7, 10, 13])
        num_people['full'] = fuzz.trimf(num_people.universe, [10, 14, 15])

        return num_people

    @staticmethod
    def __init_cooling_power_universe():
        """
        Initializes the fuzzy universe for cooling power output with categories:
        'disabled', 'very low', 'low', 'average', 'high', and 'very high'.
        """
        cooling_power = ctrl.Consequent(np.arange(0, 101, 1), 'Cooling power')

        cooling_power['disabled'] = fuzz.trimf(cooling_power.universe, [0, 0, 0])
        cooling_power['very low'] = fuzz.trimf(cooling_power.universe, [1, 1, 25])
        cooling_power['low'] = fuzz.trimf(cooling_power.universe, [10, 30, 50])
        cooling_power['average'] = fuzz.trimf(cooling_power.universe, [40, 50, 70])
        cooling_power['high'] = fuzz.trimf(cooling_power.universe, [60, 80, 90])
        cooling_power['very high'] = fuzz.trimf(cooling_power.universe, [85, 100, 100])

        return cooling_power

    def __generate_rules(self):
        """
        Defines fuzzy rules to control cooling power based on input conditions.
        """
        return [
            ctrl.Rule(self.num_people['empty'], self.cooling_power['disabled']),

            ctrl.Rule(
                (self.temp_inside['cool'] | (self.temp_inside['cool'] & (
                        self.temp_outside['very cold'] | self.temp_outside[
                    'cold']))) & self.__is_not_empty_room_rule(),
                self.cooling_power['disabled']
            ),

            ctrl.Rule(self.temp_inside['comfort'] & self.__is_not_empty_room_rule(), self.cooling_power['very low']),

            ctrl.Rule(self.temp_inside['comfort'] & (
                    self.num_people['many'] | self.num_people['full']) & self.__is_not_empty_room_rule(),
                      self.cooling_power['low']),

            ctrl.Rule(self.temp_inside['hot'] & self.__is_not_empty_room_rule(), self.cooling_power['average']),

            ctrl.Rule(self.temp_inside['hot'] & (self.num_people['many'] | self.num_people['full']),
                      self.cooling_power['high']),

            ctrl.Rule(self.temp_inside['very hot'] & self.__is_not_empty_room_rule(),
                      self.cooling_power['very high']),
            ctrl.Rule(
                self.temp_inside['hot'] & self.temp_outside['very warm'] & (
                        self.num_people['many'] | self.num_people['full']),
                self.cooling_power['very high']),
        ]

    def __is_not_empty_room_rule(self):
        """
        Helper function to determine if the room is not empty based on occupancy levels.
        """
        return self.num_people['few'] | self.num_people['medium'] | self.num_people['many'] | self.num_people['full']
