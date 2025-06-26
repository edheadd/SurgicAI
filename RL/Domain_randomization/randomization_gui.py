from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QGridLayout

class ToggleApp(QWidget):
    def __init__(self, DomainRandomizationCallbackInstance):
        super().__init__()
        
        self.DomainRandomizationCallbackInstance = DomainRandomizationCallbackInstance
        
        self.setWindowTitle('Domain Randomization Toggle')

        layout = QGridLayout()

        self.labels = []
        self.buttons = []
        name_list = DomainRandomizationCallbackInstance.name_list

        for i in range(len(name_list)):
            label = QLabel(name_list[i])
            button = QPushButton()
            if DomainRandomizationCallbackInstance.randomization_params[i]:
                button.setText('on')
            else:
                button.setText('off')
                
            button.clicked.connect(self.make_toggle_func(button, i))

            self.labels.append(label)
            self.buttons.append(button)
            
            layout.addWidget(label, i, 0)
            layout.addWidget(button, i, 1)

        self.setLayout(layout)

    def make_toggle_func(self, button, index):
        def toggle():
            new_state = 'off' if button.text() == 'on' else 'on'
            button.setText(new_state)
            self.DomainRandomizationCallbackInstance.update_randomization_params(index)
            print(f'Toggled {button.text()} for {self.DomainRandomizationCallbackInstance.name_list[index]}, change will occur next reset')
        return toggle

