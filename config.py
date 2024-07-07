import configparser

class Config:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.contract_id = config['DEFAULT']['CONTRACT_ID']
        self.storage_wanted_img = config['DEFAULT']['AWS_BUCKET_WANTED_IMG']
        self.url_facial_api = config['DEFAULT']['URL_FACIAL_API']
        self.token_facial_api = config['DEFAULT']['TOKEN_FACIAL_API']
        self.aws_access_key_id = config['DEFAULT']['AWS_ACCESS_KEY_ID']
        self.aws_secret_access_key = config['DEFAULT']['AWS_SECRET_ACCESS_KEY']
