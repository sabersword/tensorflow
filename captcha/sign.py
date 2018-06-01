import requests
import tensorflow as tf
import captcha_recognize
import sys
import time

LOGIN_FAIL_LIMIT = 10
CAPTCHA_FAIL_LIMIT = 10
tf_session = tf.InteractiveSession()


def main(username, password, sign_type):
    login_fail_times = 0
    captcha_fail_times = 0
    # 登录
    while True:
        http_session = requests.Session()
        requests.get('http://itable.abc/')

        login_data = {'AuthenticationType': '0', 'RegisterName': username, 'Token': password}
        login_resp = http_session.post('http://itable.abc/iTable/login/LoginAction_login.action', data=login_data)
        # print(login_resp.text)

        itable_resp = http_session.get('http://itable.abc/iTable/applyST.action?resourceID=iTableAttend&appID=e8b543c9f91e4a26826e40f813ab323d')
        # print(itable_resp.status_code)
        # print(itable_resp.headers)
        # print(itable_resp.text)
        if 'javascript' in itable_resp.text:
            break
        login_fail_times += 1
        if login_fail_times > LOGIN_FAIL_LIMIT:
            sys.exit(1)

    # 获取验证码,并签到签退
    while True:
        img = http_session.get('http://10.229.134.170/iTableAttendance/userAttendance/UserRecordAction_generateCode.action?time=' + str(round(time.time() * 1000)))
        # print(img.status_code)
        # print(img.headers)
        if 'jpeg' not in img.headers['Content-Type']:
            captcha_fail_times += 1
            if captcha_fail_times > CAPTCHA_FAIL_LIMIT:
                sys.exit(1)

        # 识别验证码
        # with open('E:/picture_test/' + '11' + '.jpg', 'wb') as file:
        #     file.write(img.content)
        img_data = tf.image.decode_jpeg(img.content).eval() / 255
        captcha_code = captcha_recognize.recognize(img_data)
        print(captcha_code)
        sign_data = {'verifyCode': captcha_code}
        if sign_type == 'sign_out':
            sign_url = 'http://10.229.134.170/iTableAttendance/userAttendance/UserRecordAction_signOut.action'
        else:
            sign_url = 'http://10.229.134.170/iTableAttendance/userAttendance/UserRecordAction_signIn.action'
        sign_resp = http_session.post(sign_url, data=sign_data)
        # print(sign_resp.text)
        if 'success' not in sign_resp.text and '今天已经有签到记录' not in sign_resp.text:
            captcha_fail_times += 1
            if captcha_fail_times > CAPTCHA_FAIL_LIMIT:
                sys.exit(1)
        else:  # 成功签到or签退
            print('成功签到or签退')
            sys.exit(0)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise Exception
    main(sign_type=sys.argv[3], username=sys.argv[1], password=sys.argv[2])
