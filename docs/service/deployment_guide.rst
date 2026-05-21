Service Deployment Guide
========================

This page is intended to help provide a reference guide for devops engineers deploying the Service. NONE of this is necessary if you are only writing Client scripts, please see the `Dataclass Guide <../dial_dataclass/overview.html>`_ if that is the case.

NOTE: You will need to either `install MongoDB <https://www.mongodb.com/docs/manual/installation/>`_ or have the credentials needed to connect to another service before you are able to run the INTERSECT-DIAL Service.

From source
===========

1. Clone the repository and cd into it.
2. Install `uv` and run `uv sync --no-dev`
3. Run `uv run scripts/launch_service.py --config path/to/config.json`

From Docker
===========

`ghcr.io/intersect-dial/dial:latest` will get you the latest TAGGED version. These are meant to be the most stable releases. You can also replace the `latest` tag with the appropriate version tag.

`ghcr.io/intersect-dial/dial:main` will get you the version which matches our stable deployments. Note that this image will probably be less stable than the tagged version, and may or may not be behind it.

`ghcr.io/intersect-dial/dial:develop` will get you the bleeding edge version. This image is intended to quickly consolidate various features into an active branch. Don't use this image if you're not prepared for things to break.

For a full reference, see https://github.com/INTERSECT-DIAL/dial/pkgs/container/dial . The Dockerfile source code lives at https://github.com/INTERSECT-DIAL/dial/blob/develop/Dockerfile

From the Helm Chart
===================

We provide a Helm chart, which can be accessed at https://intersect-dial.github.io/dial/index.yaml . To view the source code for the charts, see https://github.com/INTERSECT-DIAL/dial/tree/develop/charts

Config File Reference
=====================

An example configuration file looks like this:

.. literalinclude:: ../../local-conf.json

All variables under the `intersect-hierarchy` namespace are used to define the URI for you. All variables under `intersect` define the credentials for connecting to the control plane + data plane. Unless you are deploying an entire INTERSECT ecosystem yourself, you will generally not need to configure the broker yourself. In the future, the broker credentials
NOTE: This is subject to change in the near future with the INTERSECT Registry Service, where you will need to manually register your service against an authentication server, but will provide far fewer configuration variables. In the future, the control plane + data plane credentials will be managed server-side, and INTERSECT will handle the management of them automatically.

Under `dial: mongo` , you will need to provide the credentials to the Mongo database you are using. The fields listed in this configuration file are the defaults; you do not have to provide the individual lines.

`rosenbrock_hierarchy` is meant to deploy `scripts/mock_rosenbrock_service.py`. This is unnecessary for the final deployment, but can be useful in debugging your INTERSECT setup.
