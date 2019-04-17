<?php

$x = exec("python examples/task1Api.py b085_110_120.wav");
file_put_contents("x.txt", $x);