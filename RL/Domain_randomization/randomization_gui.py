from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QGridLayout

class GUI(QWidget):
    def __init__(self, DomainRandomizationCallbackInstance):
        super().__init__()
        
        self.func_dict = DomainRandomizationCallbackInstance.func_dict
        self.name_list = list(self.func_dict.keys())
        self.reset_env = DomainRandomizationCallbackInstance.reset_env
        self.update_randomization_params = DomainRandomizationCallbackInstance.update_randomization_params
        
        self.setWindowTitle('Domain Randomization Toggle')
        
        self.immediate_randomization = False

        layout = QGridLayout()

        for i in range(len(self.name_list)):
            property_dict = self.func_dict[self.name_list[i]]
            label = QLabel(self.name_list[i])
            button = QPushButton()
            description_label = QLabel(property_dict.get("description", "No description provided"))
            if property_dict["status"]:
                button.setText('on')
            else:
                button.setText('off')
                
            button.clicked.connect(self.make_toggle_func(button, self.name_list[i]))
            
            layout.addWidget(label, i, 0)
            layout.addWidget(button, i, 1)
            layout.addWidget(description_label, i, 2)
            
            
        immediate_label = QLabel("Immediate Randomization")    
        immediate_info = QLabel("If on, randomization will take effect on toggle, otherwise it will take effect after reset")
        immediate_button = QPushButton()
        immediate_button.setText('off')            
        immediate_button.clicked.connect(self.make_immediate_toggle_func(immediate_button))
        layout.addWidget(immediate_label, len(self.name_list)+1, 0)
        layout.addWidget(immediate_button, len(self.name_list)+1, 1)
        layout.addWidget(immediate_info, len(self.name_list)+1, 2)
        
        reset_label = QLabel("Reset")
        reset_info = QLabel("On press, resets the environment and randomizes the parameters per the current state of the toggles")
        reset_button = QPushButton()
        reset_button.setText('Reset')           
        reset_button.clicked.connect(self.reset_env)
        layout.addWidget(reset_label, len(self.name_list)+2, 0)
        layout.addWidget(reset_button, len(self.name_list)+2, 1)
        layout.addWidget(reset_info, len(self.name_list)+2, 2)
        
        self.setLayout(layout)

    def make_toggle_func(self, button, name):
        def toggle():
            new_state = 'off' if button.text() == 'on' else 'on'
            button.setText(new_state)
            self.update_randomization_params(name, self.immediate_randomization)
            print(f'Toggled {button.text()} for {name}')
        return toggle
    
    def make_immediate_toggle_func(self, button):
        def toggle():
            new_state = 'off' if button.text() == 'on' else 'on'
            button.setText(new_state)
            self.immediate_randomization = not self.immediate_randomization
            print(f'Immediate randomization set to {self.immediate_randomization}')
        return toggle

