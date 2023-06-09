# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six

from lightly.openapi_generated.swagger_client.api_client import ApiClient


class CollaborationApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def create_or_update_shared_access_config_by_dataset_id(self, body, dataset_id, **kwargs):  # noqa: E501
        """create_or_update_shared_access_config_by_dataset_id  # noqa: E501

        Create or update a shared access config.  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_or_update_shared_access_config_by_dataset_id(body, dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param SharedAccessConfigCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.create_or_update_shared_access_config_by_dataset_id_with_http_info(body, dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.create_or_update_shared_access_config_by_dataset_id_with_http_info(body, dataset_id, **kwargs)  # noqa: E501
            return data

    def create_or_update_shared_access_config_by_dataset_id_with_http_info(self, body, dataset_id, **kwargs):  # noqa: E501
        """create_or_update_shared_access_config_by_dataset_id  # noqa: E501

        Create or update a shared access config.  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_or_update_shared_access_config_by_dataset_id_with_http_info(body, dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param SharedAccessConfigCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body', 'dataset_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method create_or_update_shared_access_config_by_dataset_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if self.api_client.client_side_validation and ('body' not in params or
                                                       params['body'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `body` when calling `create_or_update_shared_access_config_by_dataset_id`")  # noqa: E501
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `create_or_update_shared_access_config_by_dataset_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'body' in params:
            body_params = params['body']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth', 'auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/collaboration/access', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='CreateEntityResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def delete_shared_access_config_by_id(self, dataset_id, access_config_id, **kwargs):  # noqa: E501
        """delete_shared_access_config_by_id  # noqa: E501

        Delete shared access config by id.  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_shared_access_config_by_id(dataset_id, access_config_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID access_config_id: ObjectId of the shared access config. (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.delete_shared_access_config_by_id_with_http_info(dataset_id, access_config_id, **kwargs)  # noqa: E501
        else:
            (data) = self.delete_shared_access_config_by_id_with_http_info(dataset_id, access_config_id, **kwargs)  # noqa: E501
            return data

    def delete_shared_access_config_by_id_with_http_info(self, dataset_id, access_config_id, **kwargs):  # noqa: E501
        """delete_shared_access_config_by_id  # noqa: E501

        Delete shared access config by id.  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.delete_shared_access_config_by_id_with_http_info(dataset_id, access_config_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID access_config_id: ObjectId of the shared access config. (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'access_config_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method delete_shared_access_config_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `delete_shared_access_config_by_id`")  # noqa: E501
        # verify the required parameter 'access_config_id' is set
        if self.api_client.client_side_validation and ('access_config_id' not in params or
                                                       params['access_config_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `access_config_id` when calling `delete_shared_access_config_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'access_config_id' in params:
            path_params['accessConfigId'] = params['access_config_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth', 'auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/collaboration/access/{accessConfigId}', 'DELETE',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type=None,  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_shared_access_configs_by_dataset_id(self, dataset_id, **kwargs):  # noqa: E501
        """get_shared_access_configs_by_dataset_id  # noqa: E501

        Get shared access configs by datasetId.  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_shared_access_configs_by_dataset_id(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: list[SharedAccessConfigData]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_shared_access_configs_by_dataset_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_shared_access_configs_by_dataset_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
            return data

    def get_shared_access_configs_by_dataset_id_with_http_info(self, dataset_id, **kwargs):  # noqa: E501
        """get_shared_access_configs_by_dataset_id  # noqa: E501

        Get shared access configs by datasetId.  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_shared_access_configs_by_dataset_id_with_http_info(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: list[SharedAccessConfigData]
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_shared_access_configs_by_dataset_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_shared_access_configs_by_dataset_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth', 'auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/collaboration/access', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='list[SharedAccessConfigData]',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)
