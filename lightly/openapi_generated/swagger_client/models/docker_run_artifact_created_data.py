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


class DockerRunArtifactCreatedData(object):
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
        'signed_write_url': 'str',
        'artifact_id': 'str'
    }

    attribute_map = {
        'signed_write_url': 'signedWriteUrl',
        'artifact_id': 'artifactId'
    }

    def __init__(self, signed_write_url=None, artifact_id=None, _configuration=None):  # noqa: E501
        """DockerRunArtifactCreatedData - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._signed_write_url = None
        self._artifact_id = None
        self.discriminator = None

        self.signed_write_url = signed_write_url
        self.artifact_id = artifact_id

    @property
    def signed_write_url(self):
        """Gets the signed_write_url of this DockerRunArtifactCreatedData.  # noqa: E501


        :return: The signed_write_url of this DockerRunArtifactCreatedData.  # noqa: E501
        :rtype: str
        """
        return self._signed_write_url

    @signed_write_url.setter
    def signed_write_url(self, signed_write_url):
        """Sets the signed_write_url of this DockerRunArtifactCreatedData.


        :param signed_write_url: The signed_write_url of this DockerRunArtifactCreatedData.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and signed_write_url is None:
            raise ValueError("Invalid value for `signed_write_url`, must not be `None`")  # noqa: E501

        self._signed_write_url = signed_write_url

    @property
    def artifact_id(self):
        """Gets the artifact_id of this DockerRunArtifactCreatedData.  # noqa: E501


        :return: The artifact_id of this DockerRunArtifactCreatedData.  # noqa: E501
        :rtype: str
        """
        return self._artifact_id

    @artifact_id.setter
    def artifact_id(self, artifact_id):
        """Sets the artifact_id of this DockerRunArtifactCreatedData.


        :param artifact_id: The artifact_id of this DockerRunArtifactCreatedData.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and artifact_id is None:
            raise ValueError("Invalid value for `artifact_id`, must not be `None`")  # noqa: E501

        self._artifact_id = artifact_id

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
        if issubclass(DockerRunArtifactCreatedData, dict):
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
        if not isinstance(other, DockerRunArtifactCreatedData):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, DockerRunArtifactCreatedData):
            return True

        return self.to_dict() != other.to_dict()
