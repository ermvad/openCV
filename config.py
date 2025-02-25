DARK_STYLE = """
QWidget {
    background-color: #2e2e2e;
    color: #f0f0f0;
    font-family: Arial;
    font-size: 14px;
}
QPushButton {
    background-color: #3e3e3e;
    border: 1px solid #555;
    border-radius: 5px;
    padding: 5px;
}
QPushButton:hover {
    background-color: #505050;
}
QLabel {
    color: #f0f0f0;
}
QTabWidget::pane { 
    border: 1px solid #555; 
    background: #3e3e3e;
}
QTabBar::tab {
    background: #3e3e3e;
    border: 1px solid #555;
    padding: 5px;
    margin: 2px;
}
QTabBar::tab:selected {
    background: #505050;
    border: 1px solid #777;
}
QFileDialog {
    background-color: #2e2e2e;
    color: #f0f0f0;
}
QScrollArea {
    background-color: #2e2e2e;
}
QGroupBox {
    border: 1px solid #555;
    border-radius: 5px;
    margin-top: 2ex;
    padding-top: 1ex;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
}
QComboBox {
    background-color: #3e3e3e;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px;
}
QComboBox::drop-down {
    border-left: 1px solid #555;
}
QComboBox QAbstractItemView {
    background-color: #3e3e3e;
    border: 1px solid #555;
    selection-background-color: #505050;
    selection-color: #f0f0f0;
}
QLineEdit {
    background-color: #3e3e3e;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px;
}
QCheckBox {
    spacing: 5px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid #555;
    background: #3e3e3e;
}
QCheckBox::indicator:checked {
    border: 1px solid #8f8;
    background: #5a5;
}
QSpinBox, QDoubleSpinBox {
    background-color: #3e3e3e;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px;
}
QListWidget {
    background-color: #3e3e3e;
    border: 1px solid #555;
    border-radius: 3px;
    alternate-background-color: #353535;
}
QListWidget::item:selected {
    background-color: #505050;
}
"""

# глобальные настройки приложения
APP_NAME = "OpenCV Filters"
APP_VERSION = "1.0.0"