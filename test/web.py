def application(environ, start_response):

    import mysql.connector
    db = mysql.connector.connect(host="localhost", user="root", passwd="root", database='dbgirl')
    cursor = db.cursor()
    cursor.execute("SELECT * FROM user")
    rows = cursor.fetchall()
    start_response('200 OK', [('Content-Type', 'text/html')])
    body = '%s' % rows

    return [body.encode('utf-8')]