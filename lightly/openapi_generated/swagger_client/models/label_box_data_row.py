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


class LabelBoxDataRow(object):
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
        'external_id': 'str',
        'image_url': 'RedirectedReadUrl'
    }

    attribute_map = {
        'external_id': 'externalId',
        'image_url': 'imageUrl'
    }

    def __init__(self, external_id=None, image_url=None, _configuration=None):  # noqa: E501
        """LabelBoxDataRow - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._external_id = None
        self._image_url = None
        self.discriminator = None

        self.external_id = external_id
        self.image_url = image_url

    @property
    def external_id(self):
        """Gets the external_id of this LabelBoxDataRow.  # noqa: E501

        The task_id for importing into LabelBox.  # noqa: E501

        :return: The external_id of this LabelBoxDataRow.  # noqa: E501
        :rtype: str
        """
        return self._external_id

    @external_id.setter
    def external_id(self, external_id):
        """Sets the external_id of this LabelBoxDataRow.

        The task_id for importing into LabelBox.  # noqa: E501

        :param external_id: The external_id of this LabelBoxDataRow.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and external_id is None:
            raise ValueError("Invalid value for `external_id`, must not be `None`")  # noqa: E501

        self._external_id = external_id

    @property
    def image_url(self):
        """Gets the image_url of this LabelBoxDataRow.  # noqa: E501


        :return: The image_url of this LabelBoxDataRow.  # noqa: E501
        :rtype: RedirectedReadUrl
        """
        return self._image_url

    @image_url.setter
    def image_url(self, image_url):
        """Sets the image_url of this LabelBoxDataRow.


        :param image_url: The image_url of this LabelBoxDataRow.  # noqa: E501
        :type: RedirectedReadUrl
        """
        if self._configuration.client_side_validation and image_url is None:
            raise ValueError("Invalid value for `image_url`, must not be `None`")  # noqa: E501

        self._image_url = image_url

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
        if issubclass(LabelBoxDataRow, dict):
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
        if not isinstance(other, LabelBoxDataRow):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, LabelBoxDataRow):
            return True

        return self.to_dict() != other.to_dict()
