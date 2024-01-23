+-----------------+
| Start           |
+-----------------+
        |
        v
+-----------------------------------+
| Call query() with payload         |
| {"inputs": "Today is a sunny..."} |
+-----------------------------------+
        |
        v
+-----------------------------+
| Send POST request to API    |
| (URL: https://api-inference...|
|  Headers: Authorization     |
|           Bearer token      |
|  Method: POST               |
|  Body: JSON.stringify(data) |
+-----------------------------+
        |
        v
+-------------------------+
| Wait for API response   |
| (result of fetch())     |
+-------------------------+
        |
        v
+----------------------------+
| Parse response as JSON    |
| (response.json())         |
+----------------------------+
        |
        v
+-----------------------------+
| Return parsed JSON result   |
+-----------------------------+
        |
        v
+--------------------------------------+
| Handle returned promise               |
| (then() callback)                     |
|  Log response to console              |
|  (JSON.stringify(response))           |
+--------------------------------------+
        |
        v
+-----------------+
| End             |
+-----------------+
