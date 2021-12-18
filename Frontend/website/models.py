from enum import unique
from . import db #from the current package import db
from flask_login import UserMixin# gives user model things specific to it
from sqlalchemy.sql import func

#In python the model classes are capitalized by convention, 
#however the sql table name will be lowercase.
class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)#optional will automatically be created
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))# foreign key references that references user column(one to many relationship)
   # Look up other db relationships
class User(db.Model, UserMixin):# the model to inherite from model and UserMixin
    id = db.Column(db.Integer, primary_key = True)
    email = db.Column(db.String(150), unique = True)#column to store string values of length 150 and unique
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    notes = db.relationship('Note')# contains notes of the user 
    # relationship refers to the model (capitalized)
    #foreign key refers to the db table