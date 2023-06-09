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


class SamaTaskData(object):
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
        'id': 'int',
        'url': 'RedirectedReadUrl',
        'image': 'RedirectedReadUrl',
        'lightly_file_name': 'str',
        'lightly_meta_info': 'str'
    }

    attribute_map = {
        'id': 'id',
        'url': 'url',
        'image': 'image',
        'lightly_file_name': 'lightlyFileName',
        'lightly_meta_info': 'lightlyMetaInfo'
    }

    def __init__(self, id=None, url=None, image=None, lightly_file_name=None, lightly_meta_info=None, _configuration=None):  # noqa: E501
        """SamaTaskData - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._id = None
        self._url = None
        self._image = None
        self._lightly_file_name = None
        self._lightly_meta_info = None
        self.discriminator = None

        self.id = id
        self.url = url
        if image is not None:
            self.image = image
        if lightly_file_name is not None:
            self.lightly_file_name = lightly_file_name
        if lightly_meta_info is not None:
            self.lightly_meta_info = lightly_meta_info

    @property
    def id(self):
        """Gets the id of this SamaTaskData.  # noqa: E501


        :return: The id of this SamaTaskData.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this SamaTaskData.


        :param id: The id of this SamaTaskData.  # noqa: E501
        :type: int
        """
        if self._configuration.client_side_validation and id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def url(self):
        """Gets the url of this SamaTaskData.  # noqa: E501


        :return: The url of this SamaTaskData.  # noqa: E501
        :rtype: RedirectedReadUrl
        """
        return self._url

    @url.setter
    def url(self, url):
        """Sets the url of this SamaTaskData.


        :param url: The url of this SamaTaskData.  # noqa: E501
        :type: RedirectedReadUrl
        """
        if self._configuration.client_side_validation and url is None:
            raise ValueError("Invalid value for `url`, must not be `None`")  # noqa: E501

        self._url = url

    @property
    def image(self):
        """Gets the image of this SamaTaskData.  # noqa: E501


        :return: The image of this SamaTaskData.  # noqa: E501
        :rtype: RedirectedReadUrl
        """
        return self._image

    @image.setter
    def image(self, image):
        """Sets the image of this SamaTaskData.


        :param image: The image of this SamaTaskData.  # noqa: E501
        :type: RedirectedReadUrl
        """

        self._image = image

    @property
    def lightly_file_name(self):
        """Gets the lightly_file_name of this SamaTaskData.  # noqa: E501

        The original fileName of the sample. This is unique within a dataset  # noqa: E501

        :return: The lightly_file_name of this SamaTaskData.  # noqa: E501
        :rtype: str
        """
        return self._lightly_file_name

    @lightly_file_name.setter
    def lightly_file_name(self, lightly_file_name):
        """Sets the lightly_file_name of this SamaTaskData.

        The original fileName of the sample. This is unique within a dataset  # noqa: E501

        :param lightly_file_name: The lightly_file_name of this SamaTaskData.  # noqa: E501
        :type: str
        """

        self._lightly_file_name = lightly_file_name

    @property
    def lightly_meta_info(self):
        """Gets the lightly_meta_info of this SamaTaskData.  # noqa: E501


        :return: The lightly_meta_info of this SamaTaskData.  # noqa: E501
        :rtype: str
        """
        return self._lightly_meta_info

    @lightly_meta_info.setter
    def lightly_meta_info(self, lightly_meta_info):
        """Sets the lightly_meta_info of this SamaTaskData.


        :param lightly_meta_info: The lightly_meta_info of this SamaTaskData.  # noqa: E501
        :type: str
        """

        self._lightly_meta_info = lightly_meta_info

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
        if issubclass(SamaTaskData, dict):
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
        if not isinstance(other, SamaTaskData):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, SamaTaskData):
            return True

        return self.to_dict() != other.to_dict()
