from flask import Flask, render_template, request, redirect, url_for
from model.utils import get_city_weather, predict_flood