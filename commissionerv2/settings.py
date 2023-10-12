"""
Django settings for commissionerv2 project.

Generated by 'django-admin startproject' using Django 4.2.5.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
from os import path, environ
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!

SECRET_KEY = environ.get("SECRET_KEY")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = int(environ.get("DEBUG", default=0))
ALLOWED_HOSTS = environ.get("DJANGO_ALLOWED_HOSTS").split(" ") + ['10.0.2.2']
CSRF_TRUSTED_ORIGINS = environ.get("DJANGO_CSRF_ALLOWED_HOSTS").split(" ") + ['http://172.20.10.14:1337']


AUTH_USER_MODEL = 'traders.Trader'


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'knox',
    'storages',
    'mapwidgets',
    'django_extensions',
    ##
    'apps.traders',
    'apps.environment_simulator',
    'apps.scraper_pipes'
]

REST_KNOX = {
    'TOKEN_TTL': timedelta(hours=240),
}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'commissionerv2.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'commissionerv2.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': environ.get("POSTGRES_DB"),
        'USER': environ.get("POSTGRES_USER"),
        'PASSWORD': environ.get("POSTGRES_PASSWORD"),
        'HOST': environ.get("DB_HOST"),
        'PORT': environ.get("DB_PORT"),
    }
}

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

if environ.get("OS_ENV") == "PROD":
    AWS_PRELOAD_METADATA = True
    DEFAULT_FILE_STORAGE = 'general_funcs.myStorage.MediaStorage'
    STATICFILES_STORAGE = 'general_funcs.myStorage.StaticStorage'
    AWS_STORAGE_BUCKET_NAME = environ.get("BUCKETNAME")
    AWS_S3_REGION_NAME = environ.get("BUCKETREGION")
    AWS_S3_ENDPOINT_URL = environ.get("SPACE_ENDPOINT")
    AWS_ACCESS_KEY_ID = environ.get("BUCKETID")
    AWS_SECRET_ACCESS_KEY = environ.get("BUCKETSECRET")
    MEDIA_ROOT = path.join('commV2', 'media')
    STATIC_ROOT = path.join('commV2', 'static')
else:
    MEDIA_ROOT = path.join(BASE_DIR, 'media')
    STATIC_ROOT = path.join(BASE_DIR, 'static')
STATIC_URL = '/static/'
MEDIA_URL = '/media/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
