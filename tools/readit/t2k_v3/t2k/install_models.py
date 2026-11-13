from t2k.resources.downloader.downloader import download_models_and_resources

if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('username', help="Username for the domain 'gitlab.gruppometa.it'")
    arg_parser.add_argument('password', help="Password for the domain 'gitlab.gruppometa.it'")
    arg_parser.add_argument('-p', "--path", default="resources/t2k_res", help="Domain name")
    args = arg_parser.parse_args()

    download_models_and_resources(args.username, args.password, args.path)
