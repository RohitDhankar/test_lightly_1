# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from lightly.openapi_generated.swagger_client.configuration import Configuration


class DockerWorkerConfigV2LightlyModel(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'name': 'LightlyModelV2',
        'out_dim': 'int',
        'num_ftrs': 'int',
        'width': 'int'
    }

    attribute_map = {
        'name': 'name',
        'out_dim': 'outDim',
        'num_ftrs': 'numFtrs',
        'width': 'width'
    }

    def __init__(self, name=None, out_dim=None, num_ftrs=None, width=None, _configuration=None):  # noqa: E501
        """DockerWorkerConfigV2LightlyModel - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._name = None
        self._out_dim = None
        self._num_ftrs = None
        self._width = None
        self.discriminator = None

        if name is not None:
            self.name = name
        if out_dim is not None:
            self.out_dim = out_dim
        if num_ftrs is not None:
            self.num_ftrs = num_ftrs
        if width is not None:
            self.width = width

    @property
    def name(self):
        """Gets the name of this DockerWorkerConfigV2LightlyModel.  # noqa: E501


        :return: The name of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :rtype: LightlyModelV2
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this DockerWorkerConfigV2LightlyModel.


        :param name: The name of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :type: LightlyModelV2
        """

        self._name = name

    @property
    def out_dim(self):
        """Gets the out_dim of this DockerWorkerConfigV2LightlyModel.  # noqa: E501


        :return: The out_dim of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :rtype: int
        """
        return self._out_dim

    @out_dim.setter
    def out_dim(self, out_dim):
        """Sets the out_dim of this DockerWorkerConfigV2LightlyModel.


        :param out_dim: The out_dim of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :type: int
        """

        self._out_dim = out_dim

    @property
    def num_ftrs(self):
        """Gets the num_ftrs of this DockerWorkerConfigV2LightlyModel.  # noqa: E501


        :return: The num_ftrs of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :rtype: int
        """
        return self._num_ftrs

    @num_ftrs.setter
    def num_ftrs(self, num_ftrs):
        """Sets the num_ftrs of this DockerWorkerConfigV2LightlyModel.


        :param num_ftrs: The num_ftrs of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :type: int
        """

        self._num_ftrs = num_ftrs

    @property
    def width(self):
        """Gets the width of this DockerWorkerConfigV2LightlyModel.  # noqa: E501


        :return: The width of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :rtype: int
        """
        return self._width

    @width.setter
    def width(self, width):
        """Sets the width of this DockerWorkerConfigV2LightlyModel.


        :param width: The width of this DockerWorkerConfigV2LightlyModel.  # noqa: E501
        :type: int
        """

        self._width = width

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(DockerWorkerConfigV2LightlyModel, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, DockerWorkerConfigV2LightlyModel):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, DockerWorkerConfigV2LightlyModel):
            return True

        return self.to_dict() != other.to_dict()
