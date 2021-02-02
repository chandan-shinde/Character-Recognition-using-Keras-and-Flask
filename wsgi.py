from app import app
# To test on cmd type : uwsgi --socket 0.0.0.0:PORTNO --protocol=http -w wsgi:app
if __name__ == "__main__":
    app.run()
