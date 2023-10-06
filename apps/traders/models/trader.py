from django.db import models
from django.contrib.auth.models import (
    BaseUserManager, AbstractBaseUser
)
from django.contrib.auth.hashers import make_password

from general_funcs.storage_func import OverwriteStorage
from django.contrib.gis.db.models import PointField
# add number of contests allowed per game
class TraderManager(BaseUserManager):
    def _create_user(
        self, 
        date_of_birth, 
        gender, 
        name,
        user_name,
        email,
        password, 
        **extra_fields
    ):
        """
        Create and save a user with the given username, email, and password.
        """
        if not email:
            raise ValueError('The given email must be set')
        email = self.normalize_email(email)
        user = self.model(
            date_of_birth=date_of_birth, 
            gender=gender, 
            name=name, 
            user_name=user_name,
            email=email, 
            **extra_fields
        )
        user.password = make_password(password)
        user.save(using=self._db)
        return user

    def create_user(
        self, 
        date_of_birth, 
        gender, 
        name, 
        user_name,
        email, 
        password=None, 
        **extra_fields
    ):
        return self._create_user(
            date_of_birth, 
            gender, 
            name, 
            user_name,
            email, 
            password, 
            **extra_fields
        )

    def create_superuser(
        self, 
        date_of_birth, 
        gender, 
        name, 
        user_name,
        email, 
        password=None
    ):
        """
        Creates and saves a superuser with the given email, date of
        birth and password.
        """
        user = self.create_user(
            date_of_birth,
            gender,
            name,
            user_name,
            email,
            password=password,
        )
        user.is_admin = True
        user.save(using=self._db)
        return user

# add number of contests allowed per game

class Trader(AbstractBaseUser):
    REQUIRED_FIELDS = ['date_of_birth', 'gender', 'name', 'user_name']
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    gender = models.CharField(max_length=20,)
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )
    user_name = models.CharField(
        verbose_name='user name',
        max_length=255,
        unique=True,
    )
    user_string = models.CharField(
        max_length=255,
        unique=True,
        blank=True, 
        null=True
    )
    name = models.CharField(max_length=100)
    date_of_birth = models.DateField(blank=True, null=True)
    image = models.ImageField(
        upload_to='profile_images', blank=True, null=True, storage=OverwriteStorage())
    base64Image = models.TextField(blank=True, null=True)
    imageType = models.CharField(max_length=10, blank=True, null=True)
    is_admin = models.BooleanField(default=False)
    location = PointField(geography=True, blank=True, null=True)
    objects = TraderManager()
    digi6Code = models.CharField(max_length=6, blank=True, null=True)
    is6Code_verified = models.BooleanField(default=False)
    simulated_portfolio_value = models.FloatField(blank=True, null=True)

    USERNAME_FIELD = 'email'

    def save(self, *args, **kwargs):
        myUserString = None
        if self.user_name is not None:
            myUserString = self.user_name.replace(" ", "").upper()
        if myUserString is not None and myUserString!= self.user_string:
            self.user_string = myUserString         
        super(Trader, self).save(*args, **kwargs)

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        # Simplest possible answer: Yes, always
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        # Simplest possible answer: All admins are staff
        return self.is_admin

    def __str__(self):
        return str(self.id) + ', ' + self.name

    class Meta:
        verbose_name_plural = "       Traders"
        ordering = ['id']

    @staticmethod
    def get_deleted_user():
        return Trader.objects.get_or_create(
            email= "deleted@gmail.com",
            user_name="deleted",
            name="deleted",
            gender = "O",
            date_of_birth = "1995-01-01",      
        )[0]
