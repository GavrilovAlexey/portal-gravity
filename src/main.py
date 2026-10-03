from models import Engine
from views import MainWindow
from presenters import AppPresenter

if __name__ == "__main__":
    engine = Engine()
    main_window = MainWindow()

    app_presenter = AppPresenter(engine, main_window)
    app_presenter.mainloop()
