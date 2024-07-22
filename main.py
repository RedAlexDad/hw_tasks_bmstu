import numpy as np
import argparse

class ResistorCircuitSimulator:
    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.elements = []
        self.nodes = {}
        self.voltage_sources = []
        self.resistors = []
        self.connections = []

        self.G = None
        self.I = None
        self.V = None

    def read_input_file(self):
        with open(self.input_filename, 'r') as file:
            for line in file:
                self.elements.append(line.strip())

    def parse_elements(self):
        for element in self.elements:
            if element.startswith('R:'):
                parts = element.split()
                name = parts[0][2:]
                node1 = parts[1]
                node2 = parts[2]
                resistance = float(parts[3].split('=')[1])
                self.resistors.append((name, node1, node2, resistance))
                if node1 not in self.nodes:
                    self.nodes[node1] = len(self.nodes)
                if node2 not in self.nodes and node2 != 'gnd':
                    self.nodes[node2] = len(self.nodes)

            elif element.startswith('Vsrc:'):
                parts = element.split()
                name = parts[0][5:]
                node1 = parts[1]
                node2 = parts[2]
                voltage = float(parts[3].split('=')[1])
                self.voltage_sources.append((name, node1, node2, voltage))
                if node1 not in self.nodes:
                    self.nodes[node1] = len(self.nodes)
                if node2 not in self.nodes and node2 != 'gnd':
                    self.nodes[node2] = len(self.nodes)

            elif element.startswith('N:'):
                parts = element.split()
                node1 = parts[0][2:]  # Извлекаем первый узел из 'N:node1'
                node2 = parts[1]  # Второй узел
                self.connections.append((node1, node2))
                if node1 not in self.nodes:
                    self.nodes[node1] = len(self.nodes)
                if node2 not in self.nodes and node2 != 'gnd':
                    self.nodes[node2] = len(self.nodes)

    def build_system(self):
        num_nodes = len(self.nodes)
        num_voltage_sources = len(self.voltage_sources)
        size = num_nodes + num_voltage_sources

        self.G = np.zeros((size, size))
        self.I = np.zeros(size)

        node_index = {node: idx for node, idx in self.nodes.items()}

        for name, node1, node2, resistance in self.resistors:
            if node1 != 'gnd' and node2 != 'gnd':
                n1 = node_index[node1]
                n2 = node_index[node2]
                self.G[n1, n1] += 1.0 / resistance
                self.G[n2, n2] += 1.0 / resistance
                self.G[n1, n2] -= 1.0 / resistance
                self.G[n2, n1] -= 1.0 / resistance
            elif node1 == 'gnd':
                n2 = node_index[node2]
                self.G[n2, n2] += 1.0 / resistance
            elif node2 == 'gnd':
                n1 = node_index[node1]
                self.G[n1, n1] += 1.0 / resistance

        for node1, node2 in self.connections:
            if node1 != 'gnd' and node2 != 'gnd':
                n1 = node_index[node1]
                n2 = node_index[node2]
                self.G[n1, n1] += 1e12
                self.G[n2, n2] += 1e12
                self.G[n1, n2] -= 1e12
                self.G[n2, n1] -= 1e12

        voltage_idx = num_nodes
        for name, node1, node2, voltage in self.voltage_sources:
            if node1 != 'gnd':
                n1 = node_index[node1]
                self.G[voltage_idx, n1] = 1
                self.G[n1, voltage_idx] = 1
            if node2 != 'gnd':
                n2 = node_index[node2]
                self.G[voltage_idx, n2] = -1
                self.G[n2, voltage_idx] = -1
            self.I[voltage_idx] = voltage
            voltage_idx += 1

    def solve_system(self):
        self.V = np.linalg.solve(self.G, self.I)

    def post_process(self):
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1])

        with open(self.output_filename, 'w') as file:
            for node, idx in sorted_nodes:
                file.write(f"{node} {self.V[idx]:.1f}\n")

    def display_results(self):
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1])
        print('-'*50)
        print("Потенциалы в узлах:")
        for node, idx in sorted_nodes:
            print(f"{node}: {self.V[idx]:.1f} V")

        print('-'*50)

    def calculate_currents(self):
        node_index = {node: idx for node, idx in self.nodes.items()}
        print("Сила токи через резисторы:")
        for name, node1, node2, resistance in self.resistors:
            if node1 == 'gnd':
                voltage1 = 0
                voltage2 = self.V[node_index[node2]]
            elif node2 == 'gnd':
                voltage1 = self.V[node_index[node1]]
                voltage2 = 0
            else:
                voltage1 = self.V[node_index[node1]]
                voltage2 = self.V[node_index[node2]]

            current = (voltage1 - voltage2) / resistance
            print(f"{name}: {current:.3f} A")

        print('-'*50)

    def run(self):
        # Получение файлов и чтения
        self.read_input_file()
        # Парсинг элементов
        self.parse_elements()
        # Создание матрицы
        self.build_system()
        # Решение системы
        self.solve_system()
        # Вывод результатов
        self.post_process()
        # Печать результатов
        self.display_results()
        # Расчет и печать силы токов
        self.calculate_currents()

if __name__ == "__main__":
    # Ввод аргументов с командной строки
    parser = argparse.ArgumentParser(description='Resistor Circuit Simulator')
    parser.add_argument('input_file', type=str, help='Input file name')
    parser.add_argument('output_file', type=str, help='Output file name')
    args = parser.parse_args()

    # Создание экземпляра класса
    simulator = ResistorCircuitSimulator(args.input_file, args.output_file)
    simulator.run()
